import croniter
import eventlet
import netaddr
from neutron_lib.api import validators
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import netutils as heat_netutils
from heat.engine import constraints
class DNSNameConstraint(constraints.BaseCustomConstraint):

    def validate(self, value, context):
        try:
            heat_netutils.validate_dns_format(value)
        except ValueError as ex:
            self._error_message = "'%(value)s' not in valid format. Reason: %(reason)s" % {'value': value, 'reason': str(ex)}
            return False
        return True