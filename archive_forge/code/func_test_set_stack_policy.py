import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_set_stack_policy(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.set_stack_policy('stack-id', stack_policy_body='{}')
    self.assertDictEqual(api_response, {'SetStackPolicyResult': {'Some': 'content'}})
    self.assert_request_parameters({'Action': 'SetStackPolicy', 'ContentType': 'JSON', 'StackName': 'stack-id', 'StackPolicyBody': '{}', 'Version': '2010-05-15'})