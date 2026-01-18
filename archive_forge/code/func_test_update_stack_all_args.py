import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_update_stack_all_args(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.update_stack('stack_name', template_url='http://url', template_body=SAMPLE_TEMPLATE, parameters=[('KeyName', 'myKeyName'), ('KeyName2', '', True), ('KeyName3', '', False), ('KeyName4', None, True), ('KeyName5', 'Ignore Me', True)], tags={'TagKey': 'TagValue'}, notification_arns=['arn:notify1', 'arn:notify2'], disable_rollback=True, timeout_in_minutes=20, use_previous_template=True)
    self.assert_request_parameters({'Action': 'UpdateStack', 'ContentType': 'JSON', 'DisableRollback': 'true', 'NotificationARNs.member.1': 'arn:notify1', 'NotificationARNs.member.2': 'arn:notify2', 'Parameters.member.1.ParameterKey': 'KeyName', 'Parameters.member.1.ParameterValue': 'myKeyName', 'Parameters.member.2.ParameterKey': 'KeyName2', 'Parameters.member.2.UsePreviousValue': 'true', 'Parameters.member.3.ParameterKey': 'KeyName3', 'Parameters.member.3.ParameterValue': '', 'Parameters.member.4.UsePreviousValue': 'true', 'Parameters.member.4.ParameterKey': 'KeyName4', 'Parameters.member.5.UsePreviousValue': 'true', 'Parameters.member.5.ParameterKey': 'KeyName5', 'Tags.member.1.Key': 'TagKey', 'Tags.member.1.Value': 'TagValue', 'StackName': 'stack_name', 'Version': '2010-05-15', 'TimeoutInMinutes': 20, 'TemplateBody': SAMPLE_TEMPLATE, 'TemplateURL': 'http://url', 'UsePreviousTemplate': 'true'})