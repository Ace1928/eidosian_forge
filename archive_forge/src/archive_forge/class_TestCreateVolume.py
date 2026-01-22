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
class TestCreateVolume(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n            <CreateVolumeResponse xmlns="http://ec2.amazonaws.com/doc/2014-05-01/">\n              <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n              <volumeId>vol-1a2b3c4d</volumeId>\n              <size>80</size>\n              <snapshotId/>\n              <availabilityZone>us-east-1a</availabilityZone>\n              <status>creating</status>\n              <createTime>YYYY-MM-DDTHH:MM:SS.000Z</createTime>\n              <volumeType>standard</volumeType>\n              <encrypted>true</encrypted>\n            </CreateVolumeResponse>\n        '

    def test_create_volume(self):
        self.set_http_response(status_code=200)
        result = self.ec2.create_volume(80, 'us-east-1e', snapshot='snap-1a2b3c4d', encrypted=True)
        self.assert_request_parameters({'Action': 'CreateVolume', 'AvailabilityZone': 'us-east-1e', 'Size': 80, 'SnapshotId': 'snap-1a2b3c4d', 'Encrypted': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(result.id, 'vol-1a2b3c4d')
        self.assertTrue(result.encrypted)

    def test_create_volume_with_specify_kms(self):
        self.set_http_response(status_code=200)
        result = self.ec2.create_volume(80, 'us-east-1e', snapshot='snap-1a2b3c4d', encrypted=True, kms_key_id='arn:aws:kms:us-east-1:012345678910:key/abcd1234-a123-456a-a12b-a123b4cd56ef')
        self.assert_request_parameters({'Action': 'CreateVolume', 'AvailabilityZone': 'us-east-1e', 'Size': 80, 'SnapshotId': 'snap-1a2b3c4d', 'Encrypted': 'true', 'KmsKeyId': 'arn:aws:kms:us-east-1:012345678910:key/abcd1234-a123-456a-a12b-a123b4cd56ef'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(result.id, 'vol-1a2b3c4d')
        self.assertTrue(result.encrypted)