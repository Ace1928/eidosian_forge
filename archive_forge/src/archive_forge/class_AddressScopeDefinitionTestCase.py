from neutron_lib.api.definitions import address_scope
from neutron_lib.tests.unit.api.definitions import base
class AddressScopeDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = address_scope
    extension_resources = (address_scope.COLLECTION_NAME,)
    extension_attributes = ('ipv6_address_scope', 'ipv4_address_scope', 'address_scope_id')