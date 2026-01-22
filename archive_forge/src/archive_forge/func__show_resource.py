from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
def _show_resource(self):
    return self.client_plugin().show_ext_resource('tap_flow', self.resource_id)