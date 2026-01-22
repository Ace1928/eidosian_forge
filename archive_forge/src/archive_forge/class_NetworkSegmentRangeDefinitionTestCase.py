from neutron_lib.api.definitions import network_segment_range
from neutron_lib import constants
from neutron_lib.tests.unit.api.definitions import base
class NetworkSegmentRangeDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = network_segment_range
    extension_resources = (network_segment_range.COLLECTION_NAME,)
    extension_attributes = ('name', 'default', constants.SHARED, 'project_id', 'network_type', 'physical_network', 'minimum', 'maximum', 'used', 'available')