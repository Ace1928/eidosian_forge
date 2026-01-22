from neutron_lib.api.definitions import trunk_details
from neutron_lib.tests.unit.api.definitions import base
class TrunkDetailsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = trunk_details
    extension_resource = (trunk_details.COLLECTION_NAME,)
    extension_attributes = (trunk_details.TRUNK_DETAILS,)