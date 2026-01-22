from neutron_lib.api.definitions import port as port_def
from neutron_lib.api.definitions import port_mac_address_regenerate
from neutron_lib.tests.unit.api.definitions import base
class PortMacAddressRegenerateTestCase(base.DefinitionBaseTestCase):
    extension_module = port_mac_address_regenerate
    extension_attributes = (port_def.PORT_MAC_ADDRESS,)