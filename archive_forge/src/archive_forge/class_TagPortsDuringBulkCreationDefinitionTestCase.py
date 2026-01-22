from neutron_lib.api.definitions import tag_ports_during_bulk_creation
from neutron_lib.tests.unit.api.definitions import base
class TagPortsDuringBulkCreationDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = tag_ports_during_bulk_creation