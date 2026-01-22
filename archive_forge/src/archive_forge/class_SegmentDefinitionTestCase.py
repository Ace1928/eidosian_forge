from neutron_lib.api.definitions import segment
from neutron_lib.tests.unit.api.definitions import base
class SegmentDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = segment
    extension_resources = (segment.COLLECTION_NAME,)
    extension_attributes = ('network_id', segment.PHYSICAL_NETWORK, segment.NETWORK_TYPE, segment.SEGMENTATION_ID, segment.SEGMENT_ID)