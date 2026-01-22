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
class TestModifyInterfaceAttribute(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n<ModifyNetworkInterfaceAttributeResponse     xmlns="http://ec2.amazonaws.com/doc/2013-06-15/">\n    <requestId>657a4623-5620-4232-b03b-427e852d71cf</requestId>\n    <return>true</return>\n</ModifyNetworkInterfaceAttributeResponse>\n'

    def test_modify_description(self):
        self.set_http_response(status_code=200)
        self.ec2.modify_network_interface_attribute('id', 'description', 'foo')
        self.assert_request_parameters({'Action': 'ModifyNetworkInterfaceAttribute', 'NetworkInterfaceId': 'id', 'Description.Value': 'foo'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_modify_source_dest_check_bool(self):
        self.set_http_response(status_code=200)
        self.ec2.modify_network_interface_attribute('id', 'sourceDestCheck', True)
        self.assert_request_parameters({'Action': 'ModifyNetworkInterfaceAttribute', 'NetworkInterfaceId': 'id', 'SourceDestCheck.Value': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_modify_source_dest_check_str(self):
        self.set_http_response(status_code=200)
        self.ec2.modify_network_interface_attribute('id', 'sourceDestCheck', 'true')
        self.assert_request_parameters({'Action': 'ModifyNetworkInterfaceAttribute', 'NetworkInterfaceId': 'id', 'SourceDestCheck.Value': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_modify_source_dest_check_invalid(self):
        self.set_http_response(status_code=200)
        with self.assertRaises(ValueError):
            self.ec2.modify_network_interface_attribute('id', 'sourceDestCheck', 123)

    def test_modify_delete_on_termination_str(self):
        self.set_http_response(status_code=200)
        self.ec2.modify_network_interface_attribute('id', 'deleteOnTermination', True, attachment_id='bar')
        self.assert_request_parameters({'Action': 'ModifyNetworkInterfaceAttribute', 'NetworkInterfaceId': 'id', 'Attachment.AttachmentId': 'bar', 'Attachment.DeleteOnTermination': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_modify_delete_on_termination_bool(self):
        self.set_http_response(status_code=200)
        self.ec2.modify_network_interface_attribute('id', 'deleteOnTermination', 'false', attachment_id='bar')
        self.assert_request_parameters({'Action': 'ModifyNetworkInterfaceAttribute', 'NetworkInterfaceId': 'id', 'Attachment.AttachmentId': 'bar', 'Attachment.DeleteOnTermination': 'false'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_modify_delete_on_termination_invalid(self):
        self.set_http_response(status_code=200)
        with self.assertRaises(ValueError):
            self.ec2.modify_network_interface_attribute('id', 'deleteOnTermination', 123, attachment_id='bar')

    def test_modify_group_set_list(self):
        self.set_http_response(status_code=200)
        self.ec2.modify_network_interface_attribute('id', 'groupSet', ['sg-1', 'sg-2'])
        self.assert_request_parameters({'Action': 'ModifyNetworkInterfaceAttribute', 'NetworkInterfaceId': 'id', 'SecurityGroupId.1': 'sg-1', 'SecurityGroupId.2': 'sg-2'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_modify_group_set_invalid(self):
        self.set_http_response(status_code=200)
        with self.assertRaisesRegexp(TypeError, 'iterable'):
            self.ec2.modify_network_interface_attribute('id', 'groupSet', False)

    def test_modify_attr_invalid(self):
        self.set_http_response(status_code=200)
        with self.assertRaisesRegexp(ValueError, 'Unknown attribute'):
            self.ec2.modify_network_interface_attribute('id', 'invalid', 0)