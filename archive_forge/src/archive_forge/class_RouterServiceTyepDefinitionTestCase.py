from neutron_lib.api.definitions import routerservicetype
from neutron_lib.tests.unit.api.definitions import base
class RouterServiceTyepDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = routerservicetype
    extension_attributes = (routerservicetype.SERVICE_TYPE_ID,)