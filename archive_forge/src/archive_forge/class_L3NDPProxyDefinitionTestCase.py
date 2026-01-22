from neutron_lib.api.definitions import l3_ndp_proxy
from neutron_lib.tests.unit.api.definitions import base
class L3NDPProxyDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = l3_ndp_proxy
    extension_resources = (l3_ndp_proxy.COLLECTION_NAME,)
    extension_attributes = (l3_ndp_proxy.ID, l3_ndp_proxy.NAME, l3_ndp_proxy.PROJECT_ID, l3_ndp_proxy.ROUTER_ID, l3_ndp_proxy.PORT_ID, l3_ndp_proxy.IP_ADDRESS, l3_ndp_proxy.DESCRIPTION)