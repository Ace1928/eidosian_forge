from neutron_lib.api.definitions import l3_port_ip_change_not_allowed
from neutron_lib.tests.unit.api.definitions import base
class L3PortIPChangeNotAllowedDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = l3_port_ip_change_not_allowed