from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_startElement_without_name_autoEnableIO_returns_None(self):
    retval = self.volume_attribute.startElement('some name', None, None)
    self.assertEqual(retval, None)