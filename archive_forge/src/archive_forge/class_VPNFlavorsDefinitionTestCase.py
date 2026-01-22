from neutron_lib.api.definitions import vpn_flavors
from neutron_lib.tests.unit.api.definitions import base
class VPNFlavorsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = vpn_flavors
    extension_resources = (vpn_flavors.COLLECTION_NAME,)
    extension_attributes = (vpn_flavors.FLAVOR_ID,)