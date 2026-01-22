from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _check_execution(self, action, execution_id):
    """Check execution status.

        Returns False if in IDLE, RUNNING or PAUSED
        returns True if in SUCCESS
        raises ResourceFailure if in ERROR, CANCELLED
        raises ResourceUnknownState otherwise.
        """
    execution = self.client().executions.get(execution_id)
    LOG.debug('Mistral execution %(id)s is in state %(state)s' % {'id': execution_id, 'state': execution.state})
    if execution.state in ('IDLE', 'RUNNING', 'PAUSED'):
        return (False, {})
    if execution.state in ('SUCCESS',):
        return (True, jsonutils.loads(execution.output))
    if execution.state in ('ERROR', 'CANCELLED'):
        raise exception.ResourceFailure(exception_or_error=execution.state_info, resource=self, action=action)
    raise exception.ResourceUnknownStatus(resource_status=execution.state, result=_('Mistral execution is in unknown state.'))