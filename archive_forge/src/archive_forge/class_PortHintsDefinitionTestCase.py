from neutron_lib.api.definitions import port_hints
from neutron_lib.tests.unit.api.definitions import base
class PortHintsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = port_hints
    extension_resources = (port_hints.COLLECTION_NAME,)
    extension_attributes = (port_hints.HINTS,)