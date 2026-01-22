from datetime import datetime, timedelta
from mock import MagicMock, Mock
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
import boto.ec2
from boto.regioninfo import RegionInfo
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.ec2.connection import EC2Connection
from boto.ec2.snapshot import Snapshot
from boto.ec2.reservedinstance import ReservedInstancesConfiguration
from boto.compat import http_client
class TestTrimSnapshots(TestEC2ConnectionBase):
    """
    Test snapshot trimming functionality by ensuring that expected calls
    are made when given a known set of volume snapshots.
    """

    def _get_snapshots(self):
        """
        Generate a list of fake snapshots with names and dates.
        """
        snaps = []
        now = datetime.now()
        dates = [now, now - timedelta(days=1), now - timedelta(days=2), now - timedelta(days=7), now - timedelta(days=14), datetime(now.year, now.month, 1) - timedelta(days=28), datetime(now.year, now.month, 1) - timedelta(days=58), datetime(now.year, now.month, 1) - timedelta(days=88)]
        for date in dates:
            snap = Snapshot(self.ec2)
            snap.tags['Name'] = 'foo'
            snap.start_time = date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            snaps.append(snap)
        return snaps

    def test_trim_defaults(self):
        """
        Test trimming snapshots with the default arguments, which should
        keep all monthly backups forever. The result of this test should
        be that nothing is deleted.
        """
        orig = {'get_all_snapshots': self.ec2.get_all_snapshots, 'delete_snapshot': self.ec2.delete_snapshot}
        snaps = self._get_snapshots()
        self.ec2.get_all_snapshots = MagicMock(return_value=snaps)
        self.ec2.delete_snapshot = MagicMock()
        self.ec2.trim_snapshots()
        self.assertEqual(True, self.ec2.get_all_snapshots.called)
        self.assertEqual(False, self.ec2.delete_snapshot.called)
        self.ec2.get_all_snapshots = orig['get_all_snapshots']
        self.ec2.delete_snapshot = orig['delete_snapshot']

    def test_trim_months(self):
        """
        Test trimming monthly snapshots and ensure that older months
        get deleted properly. The result of this test should be that
        the two oldest snapshots get deleted.
        """
        orig = {'get_all_snapshots': self.ec2.get_all_snapshots, 'delete_snapshot': self.ec2.delete_snapshot}
        snaps = self._get_snapshots()
        self.ec2.get_all_snapshots = MagicMock(return_value=snaps)
        self.ec2.delete_snapshot = MagicMock()
        self.ec2.trim_snapshots(monthly_backups=1)
        self.assertEqual(True, self.ec2.get_all_snapshots.called)
        self.assertEqual(2, self.ec2.delete_snapshot.call_count)
        self.ec2.get_all_snapshots = orig['get_all_snapshots']
        self.ec2.delete_snapshot = orig['delete_snapshot']