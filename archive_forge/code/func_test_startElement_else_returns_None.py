from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
@mock.patch('boto.ec2.volume.TaggedEC2Object.startElement')
def test_startElement_else_returns_None(self, startElement):
    startElement.return_value = None
    volume = Volume()
    retval = volume.startElement('not tagSet or attachmentSet', None, None)
    self.assertEqual(retval, None)