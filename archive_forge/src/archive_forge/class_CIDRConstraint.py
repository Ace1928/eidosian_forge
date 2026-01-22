import croniter
import eventlet
import netaddr
from neutron_lib.api import validators
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import netutils as heat_netutils
from heat.engine import constraints
class CIDRConstraint(constraints.BaseCustomConstraint):

    def validate(self, value, context, template=None):
        try:
            netaddr.IPNetwork(netaddr.cidr_abbrev_to_verbose(value))
            msg = validators.validate_subnet(value)
            if msg is not None:
                self._error_message = msg
                return False
            return True
        except Exception as ex:
            self._error_message = 'Invalid net cidr %s ' % str(ex)
            return False