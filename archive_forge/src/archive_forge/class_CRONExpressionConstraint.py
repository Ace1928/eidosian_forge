import croniter
import eventlet
import netaddr
from neutron_lib.api import validators
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import netutils as heat_netutils
from heat.engine import constraints
class CRONExpressionConstraint(constraints.BaseCustomConstraint):

    def validate(self, value, context, template=None):
        if not value:
            return True
        try:
            croniter.croniter(value)
            return True
        except Exception as ex:
            self._error_message = _('Invalid CRON expression: %s') % str(ex)
        return False