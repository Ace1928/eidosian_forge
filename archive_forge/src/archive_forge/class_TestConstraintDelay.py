import croniter
import eventlet
import netaddr
from neutron_lib.api import validators
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import netutils as heat_netutils
from heat.engine import constraints
class TestConstraintDelay(constraints.BaseCustomConstraint):

    def validate_with_client(self, client, value):
        eventlet.sleep(value)