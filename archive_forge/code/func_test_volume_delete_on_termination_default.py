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
def test_volume_delete_on_termination_default(self):
    self.set_http_response(status_code=200)
    self.ec2.register_image('name', 'description', snapshot_id='snap-12345678')
    self.assert_request_parameters({'Action': 'RegisterImage', 'Name': 'name', 'Description': 'description', 'BlockDeviceMapping.1.DeviceName': None, 'BlockDeviceMapping.1.Ebs.DeleteOnTermination': 'false', 'BlockDeviceMapping.1.Ebs.SnapshotId': 'snap-12345678'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])