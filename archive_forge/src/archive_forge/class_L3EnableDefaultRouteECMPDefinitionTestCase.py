from neutron_lib.api.definitions import l3_enable_default_route_ecmp
from neutron_lib.tests.unit.api.definitions import base
class L3EnableDefaultRouteECMPDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = l3_enable_default_route_ecmp
    extension_resources = ()
    extension_attributes = (l3_enable_default_route_ecmp.ENABLE_DEFAULT_ROUTE_ECMP,)