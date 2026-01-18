import json
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from mock import Mock
from boto.sns.connection import SNSConnection
def test_publish_with_utf8_message(self):
    self.set_http_response(status_code=200)
    subject = message = u'We ♥ utf-8'.encode('utf-8')
    self.service_connection.publish('topic', message, subject)
    self.assert_request_parameters({'Action': 'Publish', 'TopicArn': 'topic', 'Subject': subject, 'Message': message}, ignore_params_values=['Version', 'ContentType'])