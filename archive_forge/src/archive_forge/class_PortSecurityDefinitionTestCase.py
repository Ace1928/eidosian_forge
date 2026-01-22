from neutron_lib.api.definitions import port_security
from neutron_lib.tests.unit.api.definitions import base
class PortSecurityDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = port_security
    extension_attributes = ('port_security_enabled',)