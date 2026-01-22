import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
class TestCloudFormationCreateStack(CloudFormationConnectionBase):

    def default_body(self):
        return json.dumps({u'CreateStackResponse': {u'CreateStackResult': {u'StackId': self.stack_id}, u'ResponseMetadata': {u'RequestId': u'1'}}}).encode('utf-8')

    def test_create_stack_has_correct_request_params(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_stack('stack_name', template_url='http://url', template_body=SAMPLE_TEMPLATE, parameters=[('KeyName', 'myKeyName')], tags={'TagKey': 'TagValue'}, notification_arns=['arn:notify1', 'arn:notify2'], disable_rollback=True, timeout_in_minutes=20, capabilities=['CAPABILITY_IAM'])
        self.assertEqual(api_response, self.stack_id)
        self.assert_request_parameters({'Action': 'CreateStack', 'Capabilities.member.1': 'CAPABILITY_IAM', 'ContentType': 'JSON', 'DisableRollback': 'true', 'NotificationARNs.member.1': 'arn:notify1', 'NotificationARNs.member.2': 'arn:notify2', 'Parameters.member.1.ParameterKey': 'KeyName', 'Parameters.member.1.ParameterValue': 'myKeyName', 'Tags.member.1.Key': 'TagKey', 'Tags.member.1.Value': 'TagValue', 'StackName': 'stack_name', 'Version': '2010-05-15', 'TimeoutInMinutes': 20, 'TemplateBody': SAMPLE_TEMPLATE, 'TemplateURL': 'http://url'})

    def test_create_stack_with_minimum_args(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_stack('stack_name')
        self.assertEqual(api_response, self.stack_id)
        self.assert_request_parameters({'Action': 'CreateStack', 'ContentType': 'JSON', 'DisableRollback': 'false', 'StackName': 'stack_name', 'Version': '2010-05-15'})

    def test_create_stack_fails(self):
        self.set_http_response(status_code=400, reason='Bad Request', body=b'{"Error": {"Code": 1, "Message": "Invalid arg."}}')
        with self.assertRaisesRegexp(self.service_connection.ResponseError, 'Invalid arg.'):
            api_response = self.service_connection.create_stack('stack_name', template_body=SAMPLE_TEMPLATE, parameters=[('KeyName', 'myKeyName')])

    def test_create_stack_fail_error(self):
        self.set_http_response(status_code=400, reason='Bad Request', body=b'{"RequestId": "abc", "Error": {"Code": 1, "Message": "Invalid arg."}}')
        try:
            api_response = self.service_connection.create_stack('stack_name', template_body=SAMPLE_TEMPLATE, parameters=[('KeyName', 'myKeyName')])
        except BotoServerError as e:
            self.assertEqual('abc', e.request_id)
            self.assertEqual(1, e.error_code)
            self.assertEqual('Invalid arg.', e.message)