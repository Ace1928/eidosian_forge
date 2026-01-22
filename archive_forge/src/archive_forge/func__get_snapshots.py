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