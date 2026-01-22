import croniter
import eventlet
import netaddr
from neutron_lib.api import validators
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import netutils as heat_netutils
from heat.engine import constraints
class IPCIDRConstraint(constraints.BaseCustomConstraint):

    def validate(self, value, context, template=None):
        try:
            if '/' in value:
                msg = validators.validate_subnet(value)
            else:
                msg = validators.validate_ip_address(value)
            if msg is not None:
                self._error_message = msg
                return False
            else:
                return True
        except Exception:
            self._error_message = '{} is not a valid IP or CIDR'.format(value)
            return False