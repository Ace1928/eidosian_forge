from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.connection import EC2Connection
from boto.ec2.ec2object import TaggedEC2Object
def test_remove_tag_no_value(self):
    self.set_http_response(status_code=200)
    taggedEC2Object = TaggedEC2Object(self.service_connection)
    taggedEC2Object.id = 'i-abcd1234'
    taggedEC2Object.tags['key1'] = 'value1'
    taggedEC2Object.tags['key2'] = 'value2'
    taggedEC2Object.remove_tag('key1')
    self.assert_request_parameters({'ResourceId.1': 'i-abcd1234', 'Action': 'DeleteTags', 'Tag.1.Key': 'key1'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(taggedEC2Object.tags, {'key2': 'value2'})