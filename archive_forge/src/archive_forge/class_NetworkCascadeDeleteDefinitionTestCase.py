from neutron_lib.api.definitions import network_cascade_delete
from neutron_lib.tests.unit.api.definitions import base
class NetworkCascadeDeleteDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = network_cascade_delete