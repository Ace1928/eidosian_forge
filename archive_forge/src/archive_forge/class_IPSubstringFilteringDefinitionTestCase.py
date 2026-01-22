from neutron_lib.api.definitions import ip_substring_port_filtering
from neutron_lib.tests.unit.api.definitions import base
class IPSubstringFilteringDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = ip_substring_port_filtering