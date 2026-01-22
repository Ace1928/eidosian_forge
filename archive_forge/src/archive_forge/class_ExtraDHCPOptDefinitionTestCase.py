from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.tests.unit.api.definitions import base
class ExtraDHCPOptDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = extra_dhcp_opt
    extension_resources = (extra_dhcp_opt.COLLECTION_NAME,)
    extension_attributes = (extra_dhcp_opt.EXTRADHCPOPTS,)