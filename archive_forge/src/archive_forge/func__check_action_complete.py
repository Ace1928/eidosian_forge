from oslo_log import log as logging
from oslo_serialization import jsonutils
import tempfile
from heat.common import auth_plugin
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import template
def _check_action_complete(self, action):
    with TempCACertFile(self.cacert) as cacert_path:
        stack = self.heat(cacert_path).stacks.get(stack_id=self.resource_id)
    if stack.action != action:
        return False
    if stack.status == self.IN_PROGRESS:
        return False
    elif stack.status == self.COMPLETE:
        return True
    elif stack.status == self.FAILED:
        raise exception.ResourceInError(resource_status=stack.stack_status, status_reason=stack.stack_status_reason)
    else:
        raise exception.ResourceUnknownStatus(resource_status=stack.stack_status, status_reason=stack.stack_status_reason)