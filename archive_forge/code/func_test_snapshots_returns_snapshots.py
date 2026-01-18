from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_snapshots_returns_snapshots(self):
    snapshot_one = Snapshot()
    snapshot_one.volume_id = 1
    snapshot_two = Snapshot()
    snapshot_two.volume_id = 2
    self.volume_one.connection = mock.Mock()
    self.volume_one.connection.get_all_snapshots.return_value = [snapshot_one, snapshot_two]
    retval = self.volume_one.snapshots()
    self.assertEqual(retval, [snapshot_one])