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
class TestGetAllNetworkInterfaces(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n<DescribeNetworkInterfacesResponse xmlns="http://ec2.amazonaws.com/    doc/2013-06-15/">\n    <requestId>fc45294c-006b-457b-bab9-012f5b3b0e40</requestId>\n     <networkInterfaceSet>\n       <item>\n         <networkInterfaceId>eni-0f62d866</networkInterfaceId>\n         <subnetId>subnet-c53c87ac</subnetId>\n         <vpcId>vpc-cc3c87a5</vpcId>\n         <availabilityZone>ap-southeast-1b</availabilityZone>\n         <description/>\n         <ownerId>053230519467</ownerId>\n         <requesterManaged>false</requesterManaged>\n         <status>in-use</status>\n         <macAddress>02:81:60:cb:27:37</macAddress>\n         <privateIpAddress>10.0.0.146</privateIpAddress>\n         <sourceDestCheck>true</sourceDestCheck>\n         <groupSet>\n           <item>\n             <groupId>sg-3f4b5653</groupId>\n             <groupName>default</groupName>\n           </item>\n         </groupSet>\n         <attachment>\n           <attachmentId>eni-attach-6537fc0c</attachmentId>\n           <instanceId>i-22197876</instanceId>\n           <instanceOwnerId>053230519467</instanceOwnerId>\n           <deviceIndex>5</deviceIndex>\n           <status>attached</status>\n           <attachTime>2012-07-01T21:45:27.000Z</attachTime>\n           <deleteOnTermination>true</deleteOnTermination>\n         </attachment>\n         <tagSet/>\n         <privateIpAddressesSet>\n           <item>\n             <privateIpAddress>10.0.0.146</privateIpAddress>\n             <primary>true</primary>\n           </item>\n           <item>\n             <privateIpAddress>10.0.0.148</privateIpAddress>\n             <primary>false</primary>\n           </item>\n           <item>\n             <privateIpAddress>10.0.0.150</privateIpAddress>\n             <primary>false</primary>\n           </item>\n         </privateIpAddressesSet>\n       </item>\n    </networkInterfaceSet>\n</DescribeNetworkInterfacesResponse>'

    def test_get_all_network_interfaces(self):
        self.set_http_response(status_code=200)
        result = self.ec2.get_all_network_interfaces(network_interface_ids=['eni-0f62d866'])
        self.assert_request_parameters({'Action': 'DescribeNetworkInterfaces', 'NetworkInterfaceId.1': 'eni-0f62d866'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 'eni-0f62d866')

    def test_attachment_has_device_index(self):
        self.set_http_response(status_code=200)
        parsed = self.ec2.get_all_network_interfaces()
        self.assertEqual(5, parsed[0].attachment.device_index)