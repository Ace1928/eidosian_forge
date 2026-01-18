import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_update_stack_fails(self):
    self.set_http_response(status_code=400, reason='Bad Request', body=b'Invalid arg.')
    with self.assertRaises(self.service_connection.ResponseError):
        api_response = self.service_connection.update_stack('stack_name', template_body=SAMPLE_TEMPLATE, parameters=[('KeyName', 'myKeyName')])