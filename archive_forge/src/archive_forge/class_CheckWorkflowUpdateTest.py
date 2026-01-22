from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.engine import check_resource
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import sync_point
from heat.engine import worker
from heat.rpc import api as rpc_api
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(check_resource, 'check_stack_complete')
@mock.patch.object(check_resource, 'propagate_check_resource')
@mock.patch.object(check_resource, 'check_resource_cleanup')
@mock.patch.object(check_resource, 'check_resource_update')
class CheckWorkflowUpdateTest(common.HeatTestCase):

    @mock.patch.object(worker_client.WorkerClient, 'check_resource', lambda *_: None)
    def setUp(self):
        super(CheckWorkflowUpdateTest, self).setUp()
        thread_group_mgr = mock.Mock()
        cfg.CONF.set_default('convergence_engine', True)
        self.worker = worker.WorkerService('host-1', 'topic-1', 'engine_id', thread_group_mgr)
        self.cr = check_resource.CheckResource(self.worker.engine_id, self.worker._rpc_client, self.worker.thread_group_mgr, mock.Mock(), {})
        self.worker._rpc_client = worker_client.WorkerClient()
        self.ctx = utils.dummy_context()
        self.stack = tools.get_stack('check_workflow_create_stack', self.ctx, template=tools.string_template_five, convergence=True)
        self.stack.converge_stack(self.stack.t)
        self.resource = self.stack['A']
        self.is_update = True
        self.graph_key = (self.resource.id, self.is_update)
        self.orig_load_method = stack.Stack.load
        stack.Stack.load = mock.Mock(return_value=self.stack)

    def tearDown(self):
        super(CheckWorkflowUpdateTest, self).tearDown()
        stack.Stack.load = self.orig_load_method

    def test_resource_not_available(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.worker.check_resource(self.ctx, 'non-existant-id', self.stack.current_traversal, {}, True, None)
        for mocked in [mock_cru, mock_crc, mock_pcr, mock_csc]:
            self.assertFalse(mocked.called)

    @mock.patch.object(worker.WorkerService, '_retrigger_replaced')
    def test_stale_traversal(self, mock_rnt, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.worker.check_resource(self.ctx, self.resource.id, 'stale-traversal', {}, True, None)
        self.assertTrue(mock_rnt.called)

    def test_is_update_traversal(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        mock_cru.assert_called_once_with(self.resource, self.resource.stack.t.id, set(), self.worker.engine_id, mock.ANY, mock.ANY)
        self.assertFalse(mock_crc.called)
        expected_calls = []
        for req, fwd in self.stack.convergence_dependencies.leaves():
            expected_calls.append(mock.call.worker.propagate_check_resource.assert_called_once_with(self.ctx, mock.ANY, mock.ANY, self.stack.current_traversal, mock.ANY, self.graph_key, {}, self.is_update))
        mock_csc.assert_called_once_with(self.ctx, mock.ANY, self.stack.current_traversal, self.resource.id, mock.ANY, True)

    @mock.patch.object(resource.Resource, 'load')
    @mock.patch.object(resource.Resource, 'make_replacement')
    @mock.patch.object(stack.Stack, 'time_remaining')
    def test_is_update_traversal_raise_update_replace(self, tr, mock_mr, mock_load, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_load.return_value = (self.resource, self.stack, self.stack)
        mock_cru.side_effect = resource.UpdateReplace
        tr.return_value = 317
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        mock_cru.assert_called_once_with(self.resource, self.resource.stack.t.id, set(), self.worker.engine_id, mock.ANY, mock.ANY)
        self.assertTrue(mock_mr.called)
        self.assertFalse(mock_crc.called)
        self.assertFalse(mock_pcr.called)
        self.assertFalse(mock_csc.called)

    @mock.patch.object(check_resource.CheckResource, '_stale_resource_needs_retry')
    @mock.patch.object(stack.Stack, 'time_remaining')
    def test_is_update_traversal_raise_update_inprogress(self, tr, mock_tsl, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_cru.side_effect = exception.UpdateInProgress
        self.worker.engine_id = 'some-thing-else'
        mock_tsl.return_value = True
        tr.return_value = 317
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        mock_cru.assert_called_once_with(self.resource, self.resource.stack.t.id, set(), self.worker.engine_id, mock.ANY, mock.ANY)
        self.assertFalse(mock_crc.called)
        self.assertFalse(mock_pcr.called)
        self.assertFalse(mock_csc.called)

    @mock.patch.object(resource.Resource, 'state_set')
    def test_stale_resource_retry(self, mock_ss, mock_cru, mock_crc, mock_pcr, mock_csc):
        current_template_id = self.resource.current_template_id
        res = self.cr._stale_resource_needs_retry(self.ctx, self.resource, current_template_id)
        self.assertTrue(res)
        mock_ss.assert_not_called()

    @mock.patch.object(resource.Resource, 'state_set')
    def test_try_steal_lock_alive(self, mock_ss, mock_cru, mock_crc, mock_pcr, mock_csc):
        res = self.cr._stale_resource_needs_retry(self.ctx, self.resource, str(uuid.uuid4()))
        self.assertFalse(res)
        mock_ss.assert_not_called()

    @mock.patch.object(check_resource.listener_client, 'EngineListenerClient')
    @mock.patch.object(check_resource.resource_objects.Resource, 'get_obj')
    @mock.patch.object(resource.Resource, 'state_set')
    def test_try_steal_lock_dead(self, mock_ss, mock_get, mock_elc, mock_cru, mock_crc, mock_pcr, mock_csc):
        fake_res = mock.Mock()
        fake_res.engine_id = 'some-thing-else'
        mock_get.return_value = fake_res
        mock_elc.return_value.is_alive.return_value = False
        current_template_id = self.resource.current_template_id
        res = self.cr._stale_resource_needs_retry(self.ctx, self.resource, current_template_id)
        self.assertTrue(res)
        mock_ss.assert_called_once_with(self.resource.action, resource.Resource.FAILED, mock.ANY)

    @mock.patch.object(check_resource.listener_client, 'EngineListenerClient')
    @mock.patch.object(check_resource.resource_objects.Resource, 'get_obj')
    @mock.patch.object(resource.Resource, 'state_set')
    def test_try_steal_lock_not_dead(self, mock_ss, mock_get, mock_elc, mock_cru, mock_crc, mock_pcr, mock_csc):
        fake_res = mock.Mock()
        fake_res.engine_id = self.worker.engine_id
        mock_get.return_value = fake_res
        mock_elc.return_value.is_alive.return_value = True
        current_template_id = self.resource.current_template_id
        res = self.cr._stale_resource_needs_retry(self.ctx, self.resource, current_template_id)
        self.assertFalse(res)
        mock_ss.assert_not_called()

    @mock.patch.object(stack.Stack, 'rollback')
    def test_resource_update_failure_sets_stack_state_as_failed(self, mock_tr, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.stack.state_set(self.stack.UPDATE, self.stack.IN_PROGRESS, '')
        self.resource.state_set(self.resource.UPDATE, self.resource.IN_PROGRESS)
        dummy_ex = exception.ResourceNotAvailable(resource_name=self.resource.name)
        mock_cru.side_effect = exception.ResourceFailure(dummy_ex, self.resource, action=self.resource.UPDATE)
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        s = self.stack.load(self.ctx, stack_id=self.stack.id)
        self.assertEqual((s.UPDATE, s.FAILED), (s.action, s.status))
        self.assertEqual('Resource UPDATE failed: ResourceNotAvailable: resources.A: The Resource (A) is not available.', s.status_reason)

    @mock.patch.object(stack.Stack, 'rollback')
    def test_resource_cleanup_failure_sets_stack_state_as_failed(self, mock_tr, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.is_update = False
        self.stack.state_set(self.stack.UPDATE, self.stack.IN_PROGRESS, '')
        self.resource.state_set(self.resource.UPDATE, self.resource.IN_PROGRESS)
        dummy_ex = exception.ResourceNotAvailable(resource_name=self.resource.name)
        mock_crc.side_effect = exception.ResourceFailure(dummy_ex, self.resource, action=self.resource.UPDATE)
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        s = self.stack.load(self.ctx, stack_id=self.stack.id)
        self.assertEqual((s.UPDATE, s.FAILED), (s.action, s.status))
        self.assertEqual('Resource UPDATE failed: ResourceNotAvailable: resources.A: The Resource (A) is not available.', s.status_reason)

    def test_resource_update_failure_triggers_rollback_if_enabled(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_tr = self.stack.rollback = mock.Mock(return_value=None)
        self.stack.disable_rollback = False
        self.stack.store()
        dummy_ex = exception.ResourceNotAvailable(resource_name=self.resource.name)
        mock_cru.side_effect = exception.ResourceFailure(dummy_ex, self.resource, action=self.resource.UPDATE)
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        self.assertTrue(mock_tr.called)
        mock_tr.assert_called_once_with()

    def test_resource_cleanup_failure_triggers_rollback_if_enabled(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_tr = self.stack.rollback = mock.Mock(return_value=None)
        self.is_update = False
        self.stack.disable_rollback = False
        self.stack.store()
        dummy_ex = exception.ResourceNotAvailable(resource_name=self.resource.name)
        mock_crc.side_effect = exception.ResourceFailure(dummy_ex, self.resource, action=self.resource.UPDATE)
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        mock_tr.assert_called_once_with()

    @mock.patch.object(stack.Stack, 'rollback')
    def test_rollback_is_not_triggered_on_rollback_disabled_stack(self, mock_tr, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.stack.disable_rollback = True
        self.stack.store()
        dummy_ex = exception.ResourceNotAvailable(resource_name=self.resource.name)
        mock_cru.side_effect = exception.ResourceFailure(dummy_ex, self.resource, action=self.stack.CREATE)
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        self.assertFalse(mock_tr.called)

    @mock.patch.object(stack.Stack, 'rollback')
    def test_rollback_not_re_triggered_for_a_rolling_back_stack(self, mock_tr, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.stack.disable_rollback = False
        self.stack.action = self.stack.ROLLBACK
        self.stack.status = self.stack.IN_PROGRESS
        self.stack.store()
        dummy_ex = exception.ResourceNotAvailable(resource_name=self.resource.name)
        mock_cru.side_effect = exception.ResourceFailure(dummy_ex, self.resource, action=self.stack.CREATE)
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        self.assertFalse(mock_tr.called)

    def test_resource_update_failure_purges_db_for_stack_failure(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.stack.disable_rollback = True
        self.stack.store()
        self.stack.purge_db = mock.Mock()
        dummy_ex = exception.ResourceNotAvailable(resource_name=self.resource.name)
        mock_cru.side_effect = exception.ResourceFailure(dummy_ex, self.resource, action=self.resource.UPDATE)
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        self.assertTrue(self.stack.purge_db.called)

    def test_resource_cleanup_failure_purges_db_for_stack_failure(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.is_update = False
        self.stack.disable_rollback = True
        self.stack.store()
        self.stack.purge_db = mock.Mock()
        dummy_ex = exception.ResourceNotAvailable(resource_name=self.resource.name)
        mock_crc.side_effect = exception.ResourceFailure(dummy_ex, self.resource, action=self.resource.UPDATE)
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        self.assertTrue(self.stack.purge_db.called)

    @mock.patch.object(check_resource.CheckResource, 'retrigger_check_resource')
    @mock.patch.object(stack.Stack, 'load')
    def test_initiate_propagate_rsrc_retriggers_check_rsrc_on_new_stack_update(self, mock_stack_load, mock_rcr, mock_cru, mock_crc, mock_pcr, mock_csc):
        key = sync_point.make_key(self.resource.id, self.stack.current_traversal, self.is_update)
        mock_pcr.side_effect = exception.EntityNotFound(entity='Sync Point', name=key)
        updated_stack = stack.Stack(self.ctx, self.stack.name, self.stack.t, self.stack.id, current_traversal='some_newy_trvl_uuid')
        mock_stack_load.return_value = updated_stack
        self.cr._initiate_propagate_resource(self.ctx, self.resource.id, self.stack.current_traversal, self.is_update, self.resource, self.stack)
        mock_rcr.assert_called_once_with(self.ctx, self.resource.id, updated_stack)

    def test_check_stack_complete_is_invoked_for_replaced_resource(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        resC = self.stack['C']
        is_update = True
        trav_id = self.stack.current_traversal
        replacementC_id = resC.make_replacement(self.stack.t.id, set(resC.requires))
        replacementC, stack, _ = resource.Resource.load(self.ctx, replacementC_id, trav_id, is_update, {})
        self.cr._initiate_propagate_resource(self.ctx, replacementC_id, trav_id, is_update, replacementC, self.stack)
        mock_csc.assert_called_once_with(self.ctx, self.stack, trav_id, resC.id, mock.ANY, is_update)

    @mock.patch.object(sync_point, 'sync')
    def test_retrigger_check_resource(self, mock_sync, mock_cru, mock_crc, mock_pcr, mock_csc):
        resC = self.stack['C']
        expected_predecessors = {(self.stack['A'].id, True), (self.stack['B'].id, True)}
        self.cr.retrigger_check_resource(self.ctx, resC.id, self.stack)
        mock_pcr.assert_called_once_with(self.ctx, mock.ANY, resC.id, self.stack.current_traversal, mock.ANY, (resC.id, True), None, True, None)
        call_args, call_kwargs = mock_pcr.call_args
        actual_predecessors = call_args[4]
        self.assertCountEqual(expected_predecessors, actual_predecessors)

    def test_update_retrigger_check_resource_new_traversal_deletes_rsrc(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.stack._convg_deps = dependencies.Dependencies([[(1, False), (1, True)], [(2, False), None]])
        self.cr.retrigger_check_resource(self.ctx, 2, self.stack)
        mock_pcr.assert_called_once_with(self.ctx, mock.ANY, 2, self.stack.current_traversal, mock.ANY, (2, False), None, False, None)

    def test_delete_retrigger_check_resource_new_traversal_updates_rsrc(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.stack._convg_deps = dependencies.Dependencies([[(1, False), (1, True)], [(2, False), (2, True)]])
        self.cr.retrigger_check_resource(self.ctx, 2, self.stack)
        mock_pcr.assert_called_once_with(self.ctx, mock.ANY, 2, self.stack.current_traversal, mock.ANY, (2, True), None, True, None)

    @mock.patch.object(stack.Stack, 'purge_db')
    def test_handle_failure(self, mock_purgedb, mock_cru, mock_crc, mock_pcr, mock_csc):
        self.stack.mark_failed('dummy-reason')
        mock_purgedb.assert_called_once_with()
        self.assertEqual('dummy-reason', self.stack.status_reason)

    def test_handle_failure_rollback(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_tr = self.stack.rollback = mock.Mock(return_value=None)
        self.stack.disable_rollback = False
        self.stack.state_set(self.stack.UPDATE, self.stack.IN_PROGRESS, '')
        self.stack.mark_failed('dummy-reason')
        mock_tr.assert_called_once_with()

    @mock.patch.object(stack.Stack, 'purge_db')
    @mock.patch.object(stack.Stack, 'state_set')
    @mock.patch.object(check_resource.CheckResource, 'retrigger_check_resource')
    @mock.patch.object(stack.Stack, 'rollback')
    def test_handle_rsrc_failure_when_update_fails(self, mock_tr, mock_rcr, mock_ss, mock_pdb, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_ss.return_value = False
        self.cr._handle_resource_failure(self.ctx, self.is_update, self.resource.id, self.stack, 'dummy-reason')
        self.assertTrue(mock_ss.called)
        self.assertFalse(mock_rcr.called)
        self.assertFalse(mock_pdb.called)
        self.assertFalse(mock_tr.called)

    @mock.patch.object(stack.Stack, 'purge_db')
    @mock.patch.object(stack.Stack, 'state_set')
    @mock.patch.object(check_resource.CheckResource, 'retrigger_check_resource')
    @mock.patch.object(stack.Stack, 'rollback')
    def test_handle_rsrc_failure_when_update_fails_different_traversal(self, mock_tr, mock_rcr, mock_ss, mock_pdb, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_ss.return_value = False
        new_stack = tools.get_stack('check_workflow_create_stack', self.ctx, template=tools.string_template_five, convergence=True)
        new_stack.current_traversal = 'new_traversal'
        stack.Stack.load = mock.Mock(return_value=new_stack)
        self.cr._handle_resource_failure(self.ctx, self.is_update, self.resource.id, self.stack, 'dummy-reason')
        self.assertTrue(mock_rcr.called)
        self.assertTrue(mock_ss.called)
        self.assertFalse(mock_pdb.called)
        self.assertFalse(mock_tr.called)

    def test_handle_stack_timeout(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_mf = self.stack.mark_failed = mock.Mock(return_value=True)
        self.cr._handle_stack_timeout(self.ctx, self.stack)
        mock_mf.assert_called_once_with(u'Timed out')

    def test_do_check_resource_marks_stack_as_failed_if_stack_timesout(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_mf = self.stack.mark_failed = mock.Mock(return_value=True)
        mock_cru.side_effect = scheduler.Timeout(None, 60)
        self.is_update = True
        self.cr._do_check_resource(self.ctx, self.stack.current_traversal, self.stack.t, {}, self.is_update, self.resource, self.stack, {})
        mock_mf.assert_called_once_with(u'Timed out')

    @mock.patch.object(check_resource.CheckResource, '_handle_stack_timeout')
    def test_do_check_resource_ignores_timeout_for_new_update(self, mock_hst, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_cru.side_effect = scheduler.Timeout(None, 60)
        self.is_update = True
        old_traversal = self.stack.current_traversal
        self.stack.current_traversal = 'new_traversal'
        self.cr._do_check_resource(self.ctx, old_traversal, self.stack.t, {}, self.is_update, self.resource, self.stack, {})
        self.assertFalse(mock_hst.called)

    @mock.patch.object(stack.Stack, 'has_timed_out')
    @mock.patch.object(check_resource.CheckResource, '_handle_stack_timeout')
    def test_check_resource_handles_timeout(self, mock_hst, mock_to, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_to.return_value = True
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, {})
        self.assertTrue(mock_hst.called)

    def test_check_resource_does_not_propagate_on_cancel(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_cru.side_effect = check_resource.CancelOperation
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, {})
        self.assertFalse(mock_pcr.called)
        self.assertFalse(mock_csc.called)

    @mock.patch.object(resource.Resource, 'load')
    def test_requires(self, mock_load, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_load.return_value = (self.resource, self.stack, self.stack)
        res_data = {(1, True): {u'id': 5, u'name': 'A', 'attrs': {}}, (2, True): {u'id': 3, u'name': 'B', 'attrs': {}}}
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, sync_point.serialize_input_data(res_data), self.is_update, {})
        mock_cru.assert_called_once_with(self.resource, self.resource.stack.t.id, {5, 3}, self.worker.engine_id, self.stack, mock.ANY)