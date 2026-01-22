from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
def _get_client_res_type(self, object_type):
    client_plugin = self.client_plugin()
    if object_type == self.OBJECT_NETWORK:
        return client_plugin.RES_TYPE_NETWORK
    elif object_type == self.OBJECT_QOS_POLICY:
        return client_plugin.RES_TYPE_QOS_POLICY
    else:
        return object_type