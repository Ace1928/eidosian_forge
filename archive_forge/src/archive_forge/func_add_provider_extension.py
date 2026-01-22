from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import net
from heat.engine import support
@staticmethod
def add_provider_extension(props, key):
    props['provider:' + key] = props.pop(key)