from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
@mock.patch('boto.ec2.volume.TaggedEC2Object.startElement')
def test_startElement_retval_not_None_returns_correct_thing(self, startElement):
    tag_set = mock.Mock(TagSet)
    startElement.return_value = tag_set
    volume = Volume()
    retval = volume.startElement(None, None, None)
    self.assertEqual(retval, tag_set)