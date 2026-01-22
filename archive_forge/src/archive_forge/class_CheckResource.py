import eventlet.queue
import functools
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import sync_point
from heat.objects import resource as resource_objects
from heat.rpc import api as rpc_api
from heat.rpc import listener_client
class CheckResource(object):

    def __init__(self, engine_id, rpc_client, thread_group_mgr, msg_queue, input_data):
        self.engine_id = engine_id
        self._rpc_client = rpc_client
        self.thread_group_mgr = thread_group_mgr
        self.msg_queue = msg_queue
        self.input_data = input_data

    def _stale_resource_needs_retry(self, cnxt, rsrc, prev_template_id):
        """Determine whether a resource needs retrying after failure to lock.

        Return True if we need to retry the check operation because of a
        failure to acquire the lock. This can be either because the engine
        holding the lock is no longer working, or because no other engine had
        locked the resource and the data was just out of date.

        In the former case, the lock will be stolen and the resource status
        changed to FAILED.
        """
        fields = {'current_template_id', 'engine_id'}
        rs_obj = resource_objects.Resource.get_obj(cnxt, rsrc.id, refresh=True, fields=fields)
        if rs_obj.engine_id not in (None, self.engine_id):
            if not listener_client.EngineListenerClient(rs_obj.engine_id).is_alive(cnxt):
                rs_obj.update_and_save({'engine_id': None})
                status_reason = 'Worker went down during resource %s' % rsrc.action
                rsrc.state_set(rsrc.action, rsrc.FAILED, str(status_reason))
                return True
        elif rs_obj.engine_id is None and rs_obj.current_template_id == prev_template_id:
            LOG.debug('Resource id=%d stale; retrying check', rsrc.id)
            return True
        LOG.debug('Resource id=%d modified by another traversal', rsrc.id)
        return False

    def _handle_resource_failure(self, cnxt, is_update, rsrc_id, stack, failure_reason):
        failure_handled = stack.mark_failed(failure_reason)
        if not failure_handled:
            self._retrigger_new_traversal(cnxt, stack.current_traversal, is_update, stack.id, rsrc_id)

    def _retrigger_new_traversal(self, cnxt, current_traversal, is_update, stack_id, rsrc_id):
        latest_stack = parser.Stack.load(cnxt, stack_id=stack_id, force_reload=True)
        if current_traversal != latest_stack.current_traversal:
            self.retrigger_check_resource(cnxt, rsrc_id, latest_stack)

    def _handle_stack_timeout(self, cnxt, stack):
        failure_reason = u'Timed out'
        stack.mark_failed(failure_reason)

    def _handle_resource_replacement(self, cnxt, current_traversal, new_tmpl_id, requires, rsrc, stack, adopt_stack_data):
        """Create a replacement resource and trigger a check on it."""
        try:
            new_res_id = rsrc.make_replacement(new_tmpl_id, requires)
        except exception.UpdateInProgress:
            LOG.info('No replacement created - resource already locked by new traversal')
            return
        if new_res_id is None:
            LOG.info('No replacement created - new traversal already in progress')
            self._retrigger_new_traversal(cnxt, current_traversal, True, stack.id, rsrc.id)
            return
        LOG.info('Replacing resource with new id %s', new_res_id)
        rpc_data = sync_point.serialize_input_data(self.input_data)
        self._rpc_client.check_resource(cnxt, new_res_id, current_traversal, rpc_data, True, adopt_stack_data)

    def _do_check_resource(self, cnxt, current_traversal, tmpl, resource_data, is_update, rsrc, stack, adopt_stack_data):
        prev_template_id = rsrc.current_template_id
        try:
            if is_update:
                requires = set((d.primary_key for d in resource_data.values() if d is not None))
                try:
                    check_resource_update(rsrc, tmpl.id, requires, self.engine_id, stack, self.msg_queue)
                except resource.UpdateReplace:
                    self._handle_resource_replacement(cnxt, current_traversal, tmpl.id, requires, rsrc, stack, adopt_stack_data)
                    return False
            else:
                check_resource_cleanup(rsrc, tmpl.id, self.engine_id, stack.time_remaining(), self.msg_queue)
            return True
        except exception.UpdateInProgress:
            LOG.debug('Waiting for existing update to unlock resource %s', rsrc.id)
            if self._stale_resource_needs_retry(cnxt, rsrc, prev_template_id):
                rpc_data = sync_point.serialize_input_data(self.input_data)
                self._rpc_client.check_resource(cnxt, rsrc.id, current_traversal, rpc_data, is_update, adopt_stack_data)
            else:
                rsrc.handle_preempt()
        except exception.ResourceFailure as ex:
            action = ex.action or rsrc.action
            reason = 'Resource %s failed: %s' % (action, str(ex))
            self._handle_resource_failure(cnxt, is_update, rsrc.id, stack, reason)
        except scheduler.Timeout:
            self._handle_resource_failure(cnxt, is_update, rsrc.id, stack, u'Timed out')
        except CancelOperation:
            self._retrigger_new_traversal(cnxt, current_traversal, is_update, stack.id, rsrc.id)
        return False

    def retrigger_check_resource(self, cnxt, resource_id, stack):
        current_traversal = stack.current_traversal
        graph = stack.convergence_dependencies.graph()
        update_key = parser.ConvergenceNode(resource_id, True)
        key = parser.ConvergenceNode(resource_id, update_key in graph)
        LOG.info('Re-trigger resource: %s', key)
        predecessors = set(graph[key])
        try:
            propagate_check_resource(cnxt, self._rpc_client, resource_id, current_traversal, predecessors, key, None, key.is_update, None)
        except exception.EntityNotFound as e:
            if e.entity != 'Sync Point':
                raise

    def _initiate_propagate_resource(self, cnxt, resource_id, current_traversal, is_update, rsrc, stack):
        deps = stack.convergence_dependencies
        graph = deps.graph()
        graph_key = parser.ConvergenceNode(resource_id, is_update)
        if graph_key not in graph and rsrc.replaces is not None:
            graph_key = parser.ConvergenceNode(rsrc.replaces, is_update)

        def _get_input_data(req_node, input_forward_data=None):
            if req_node.is_update:
                if input_forward_data is None:
                    return rsrc.node_data().as_dict()
                else:
                    return input_forward_data
            elif req_node.rsrc_id != graph_key.rsrc_id:
                return rsrc.replaced_by if rsrc.replaced_by is not None else resource_id
            return None
        try:
            input_forward_data = None
            for req_node in sorted(deps.required_by(graph_key), key=lambda n: n.is_update):
                input_data = _get_input_data(req_node, input_forward_data)
                if req_node.is_update:
                    input_forward_data = input_data
                propagate_check_resource(cnxt, self._rpc_client, req_node.rsrc_id, current_traversal, set(graph[req_node]), graph_key, input_data, req_node.is_update, stack.adopt_stack_data)
            if is_update:
                if input_forward_data is None:
                    rsrc.clear_stored_attributes()
                else:
                    rsrc.store_attributes()
            check_stack_complete(cnxt, stack, current_traversal, graph_key.rsrc_id, deps, graph_key.is_update)
        except exception.EntityNotFound as e:
            if e.entity == 'Sync Point':
                stack = parser.Stack.load(cnxt, stack_id=rsrc.stack.id, force_reload=True)
                if current_traversal == stack.current_traversal:
                    LOG.debug('[%s] Traversal sync point missing.', current_traversal)
                    return
                self.retrigger_check_resource(cnxt, resource_id, stack)
            else:
                raise

    def check(self, cnxt, resource_id, current_traversal, resource_data, is_update, adopt_stack_data, rsrc, stack):
        """Process a node in the dependency graph.

        The node may be associated with either an update or a cleanup of its
        associated resource.
        """
        if stack.has_timed_out():
            self._handle_stack_timeout(cnxt, stack)
            return
        tmpl = stack.t
        stack.adopt_stack_data = adopt_stack_data
        stack.thread_group_mgr = self.thread_group_mgr
        try:
            check_resource_done = self._do_check_resource(cnxt, current_traversal, tmpl, resource_data, is_update, rsrc, stack, adopt_stack_data)
            if check_resource_done:
                self._initiate_propagate_resource(cnxt, resource_id, current_traversal, is_update, rsrc, stack)
        except BaseException as exc:
            with excutils.save_and_reraise_exception():
                msg = str(exc)
                LOG.exception('Unexpected exception in resource check.')
                self._handle_resource_failure(cnxt, is_update, rsrc.id, stack, msg)