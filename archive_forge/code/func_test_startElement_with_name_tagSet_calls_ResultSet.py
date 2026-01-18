from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
@mock.patch('boto.ec2.volume.TaggedEC2Object.startElement')
@mock.patch('boto.resultset.ResultSet')
def test_startElement_with_name_tagSet_calls_ResultSet(self, ResultSet, startElement):
    startElement.return_value = None
    result_set = mock.Mock(ResultSet([('item', Tag)]))
    volume = Volume()
    volume.tags = result_set
    retval = volume.startElement('tagSet', None, None)
    self.assertEqual(retval, volume.tags)