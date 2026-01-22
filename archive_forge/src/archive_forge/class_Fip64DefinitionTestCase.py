from neutron_lib.api.definitions import fip64
from neutron_lib.tests.unit.api.definitions import base
class Fip64DefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = fip64
    extension_resources = ()
    extension_attributes = ()