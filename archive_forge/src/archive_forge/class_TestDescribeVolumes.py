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
class TestDescribeVolumes(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n            <DescribeVolumesResponse xmlns="http://ec2.amazonaws.com/doc/2014-02-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <volumeSet>\n                  <item>\n                     <volumeId>vol-1a2b3c4d</volumeId>\n                     <size>80</size>\n                     <snapshotId/>\n                     <availabilityZone>us-east-1a</availabilityZone>\n                     <status>in-use</status>\n                     <createTime>YYYY-MM-DDTHH:MM:SS.SSSZ</createTime>\n                     <attachmentSet>\n                        <item>\n                           <volumeId>vol-1a2b3c4d</volumeId>\n                           <instanceId>i-1a2b3c4d</instanceId>\n                           <device>/dev/sdh</device>\n                           <status>attached</status>\n                           <attachTime>YYYY-MM-DDTHH:MM:SS.SSSZ</attachTime>\n                           <deleteOnTermination>false</deleteOnTermination>\n                        </item>\n                     </attachmentSet>\n                     <volumeType>standard</volumeType>\n                     <encrypted>true</encrypted>\n                  </item>\n                  <item>\n                     <volumeId>vol-5e6f7a8b</volumeId>\n                     <size>80</size>\n                     <snapshotId/>\n                     <availabilityZone>us-east-1a</availabilityZone>\n                     <status>in-use</status>\n                     <createTime>YYYY-MM-DDTHH:MM:SS.SSSZ</createTime>\n                     <attachmentSet>\n                        <item>\n                           <volumeId>vol-5e6f7a8b</volumeId>\n                           <instanceId>i-5e6f7a8b</instanceId>\n                           <device>/dev/sdz</device>\n                           <status>attached</status>\n                           <attachTime>YYYY-MM-DDTHH:MM:SS.SSSZ</attachTime>\n                           <deleteOnTermination>false</deleteOnTermination>\n                        </item>\n                     </attachmentSet>\n                     <volumeType>standard</volumeType>\n                     <encrypted>false</encrypted>\n                  </item>\n               </volumeSet>\n            </DescribeVolumesResponse>\n        '

    def test_get_all_volumes(self):
        self.set_http_response(status_code=200)
        result = self.ec2.get_all_volumes(volume_ids=['vol-1a2b3c4d', 'vol-5e6f7a8b'])
        self.assert_request_parameters({'Action': 'DescribeVolumes', 'VolumeId.1': 'vol-1a2b3c4d', 'VolumeId.2': 'vol-5e6f7a8b'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, 'vol-1a2b3c4d')
        self.assertTrue(result[0].encrypted)
        self.assertEqual(result[1].id, 'vol-5e6f7a8b')
        self.assertFalse(result[1].encrypted)