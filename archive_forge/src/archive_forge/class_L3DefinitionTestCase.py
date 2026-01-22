from neutron_lib.api.definitions import l3
from neutron_lib.tests.unit.api.definitions import base
class L3DefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = l3
    extension_resources = (l3.ROUTERS, l3.FLOATINGIPS)
    extension_attributes = (l3.FLOATING_IP_ADDRESS, l3.FLOATING_NETWORK_ID, l3.ROUTER_ID, l3.PORT_ID, l3.FIXED_IP_ADDRESS, l3.SUBNET_ID, l3.EXTERNAL_GW_INFO)