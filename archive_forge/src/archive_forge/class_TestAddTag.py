import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
class TestAddTag(AWSMockServiceTestCase):
    connection_class = EmrConnection

    def default_body(self):
        return b'<AddTagsResponse\n               xmlns="http://elasticmapreduce.amazonaws.com/doc/2009-03-31">\n                   <AddTagsResult/>\n                   <ResponseMetadata>\n                        <RequestId>88888888-8888-8888-8888-888888888888</RequestId>\n                   </ResponseMetadata>\n               </AddTagsResponse>\n               '

    def test_add_mix_of_tags_with_without_values(self):
        input_tags = {'FirstKey': 'One', 'SecondKey': 'Two', 'ZzzNoValue': ''}
        self.set_http_response(200)
        with self.assertRaises(TypeError):
            self.service_connection.add_tags()
        with self.assertRaises(TypeError):
            self.service_connection.add_tags('j-123')
        with self.assertRaises(AssertionError):
            self.service_connection.add_tags('j-123', [])
        response = self.service_connection.add_tags('j-123', input_tags)
        self.assertTrue(response)
        self.assert_request_parameters({'Action': 'AddTags', 'ResourceId': 'j-123', 'Tags.member.1.Key': 'FirstKey', 'Tags.member.1.Value': 'One', 'Tags.member.2.Key': 'SecondKey', 'Tags.member.2.Value': 'Two', 'Tags.member.3.Key': 'ZzzNoValue', 'Version': '2009-03-31'})