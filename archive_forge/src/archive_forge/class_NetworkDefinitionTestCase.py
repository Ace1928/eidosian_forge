from neutron_lib.api.definitions import network
from neutron_lib.tests.unit.api.definitions import base
class NetworkDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = network
    extension_attributes = ()