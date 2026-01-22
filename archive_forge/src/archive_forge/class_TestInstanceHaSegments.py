from openstack.instance_ha.v1 import _proxy
from openstack.instance_ha.v1 import host
from openstack.instance_ha.v1 import notification
from openstack.instance_ha.v1 import segment
from openstack.instance_ha.v1 import vmove
from openstack.tests.unit import test_proxy_base
class TestInstanceHaSegments(TestInstanceHaProxy):

    def test_segments(self):
        self.verify_list(self.proxy.segments, segment.Segment)

    def test_segment_get(self):
        self.verify_get(self.proxy.get_segment, segment.Segment)

    def test_segment_create(self):
        self.verify_create(self.proxy.create_segment, segment.Segment)

    def test_segment_update(self):
        self.verify_update(self.proxy.update_segment, segment.Segment)

    def test_segment_delete(self):
        self.verify_delete(self.proxy.delete_segment, segment.Segment, True)