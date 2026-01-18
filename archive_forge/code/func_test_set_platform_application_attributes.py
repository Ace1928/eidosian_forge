import json
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from mock import Mock
from boto.sns.connection import SNSConnection
def test_set_platform_application_attributes(self):
    self.set_http_response(status_code=200)
    self.service_connection.set_platform_application_attributes(platform_application_arn='arn:myapp', attributes={'PlatformPrincipal': 'a ssl certificate', 'PlatformCredential': 'a private key'})
    self.assert_request_parameters({'Action': 'SetPlatformApplicationAttributes', 'PlatformApplicationArn': 'arn:myapp', 'Attributes.entry.1.key': 'PlatformCredential', 'Attributes.entry.1.value': 'a private key', 'Attributes.entry.2.key': 'PlatformPrincipal', 'Attributes.entry.2.value': 'a ssl certificate'}, ignore_params_values=['Version', 'ContentType'])