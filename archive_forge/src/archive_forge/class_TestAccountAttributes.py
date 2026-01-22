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
class TestAccountAttributes(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n        <DescribeAccountAttributesResponse xmlns="http://ec2.amazonaws.com/doc/2012-12-01/">\n            <requestId>6d042e8a-4bc3-43e8-8265-3cbc54753f14</requestId>\n            <accountAttributeSet>\n                <item>\n                    <attributeName>vpc-max-security-groups-per-interface</attributeName>\n                    <attributeValueSet>\n                        <item>\n                            <attributeValue>5</attributeValue>\n                        </item>\n                    </attributeValueSet>\n                </item>\n                <item>\n                    <attributeName>max-instances</attributeName>\n                    <attributeValueSet>\n                        <item>\n                            <attributeValue>50</attributeValue>\n                        </item>\n                    </attributeValueSet>\n                </item>\n                <item>\n                    <attributeName>supported-platforms</attributeName>\n                    <attributeValueSet>\n                        <item>\n                            <attributeValue>EC2</attributeValue>\n                        </item>\n                        <item>\n                            <attributeValue>VPC</attributeValue>\n                        </item>\n                    </attributeValueSet>\n                </item>\n                <item>\n                    <attributeName>default-vpc</attributeName>\n                    <attributeValueSet>\n                        <item>\n                            <attributeValue>none</attributeValue>\n                        </item>\n                    </attributeValueSet>\n                </item>\n            </accountAttributeSet>\n        </DescribeAccountAttributesResponse>\n        '

    def test_describe_account_attributes(self):
        self.set_http_response(status_code=200)
        parsed = self.ec2.describe_account_attributes()
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed[0].attribute_name, 'vpc-max-security-groups-per-interface')
        self.assertEqual(parsed[0].attribute_values, ['5'])
        self.assertEqual(parsed[-1].attribute_name, 'default-vpc')
        self.assertEqual(parsed[-1].attribute_values, ['none'])