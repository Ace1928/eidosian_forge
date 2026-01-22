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
class TestSignatureAlteration(TestEC2ConnectionBase):

    def test_unchanged(self):
        self.assertEqual(self.service_connection._required_auth_capability(), ['hmac-v4'])

    def test_switched(self):
        region = RegionInfo(name='cn-north-1', endpoint='ec2.cn-north-1.amazonaws.com.cn', connection_cls=EC2Connection)
        conn = self.connection_class(aws_access_key_id='less', aws_secret_access_key='more', region=region)
        self.assertEqual(conn._required_auth_capability(), ['hmac-v4'])