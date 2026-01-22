from neutron_lib.api.definitions import expose_l3_conntrack_helper
from neutron_lib.api.definitions import l3_conntrack_helper
from neutron_lib.tests.unit.api.definitions import base
class ExposeL3ConntrackHelperDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = expose_l3_conntrack_helper
    extension_resources = (expose_l3_conntrack_helper.COLLECTION_NAME,)
    extension_attributes = (l3_conntrack_helper.COLLECTION_NAME,)