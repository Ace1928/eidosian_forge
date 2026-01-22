import croniter
import eventlet
import netaddr
from neutron_lib.api import validators
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import netutils as heat_netutils
from heat.engine import constraints
class ISO8601Constraint(constraints.BaseCustomConstraint):

    def validate(self, value, context, template=None):
        try:
            timeutils.parse_isotime(value)
        except Exception:
            return False
        else:
            return True