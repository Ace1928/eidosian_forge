from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
@mock.patch('boto.ec2.volume.TaggedEC2Object.startElement')
def test_startElement_calls_TaggedEC2Object_startElement_with_correct_args(self, startElement):
    volume = Volume()
    volume.startElement('some name', 'some attrs', None)
    startElement.assert_called_with('some name', 'some attrs', None)