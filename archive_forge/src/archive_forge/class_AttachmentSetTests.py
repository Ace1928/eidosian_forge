from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
class AttachmentSetTests(unittest.TestCase):

    def check_that_attribute_has_been_set(self, name, value, attribute):
        attachment_set = AttachmentSet()
        attachment_set.endElement(name, value, None)
        self.assertEqual(getattr(attachment_set, attribute), value)

    def test_endElement_with_name_volumeId_sets_id(self):
        return self.check_that_attribute_has_been_set('volumeId', 'some value', 'id')

    def test_endElement_with_name_instanceId_sets_instance_id(self):
        return self.check_that_attribute_has_been_set('instanceId', 1, 'instance_id')

    def test_endElement_with_name_status_sets_status(self):
        return self.check_that_attribute_has_been_set('status', 'some value', 'status')

    def test_endElement_with_name_attachTime_sets_attach_time(self):
        return self.check_that_attribute_has_been_set('attachTime', 5, 'attach_time')

    def test_endElement_with_name_device_sets_device(self):
        return self.check_that_attribute_has_been_set('device', '/dev/null', 'device')

    def test_endElement_with_other_name_sets_other_name_attribute(self):
        return self.check_that_attribute_has_been_set('someName', 'some value', 'someName')