from oslo_utils import netutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
@classmethod
def _null_gateway_ip(cls, props):
    if cls.GATEWAY_IP not in props:
        return
    if props.get(cls.GATEWAY_IP) == '':
        props[cls.GATEWAY_IP] = None