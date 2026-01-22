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