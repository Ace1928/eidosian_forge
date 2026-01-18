import json
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from mock import Mock
from boto.sns.connection import SNSConnection
def test_set_endpoint_attributes(self):
    self.set_http_response(status_code=200)
    self.service_connection.set_endpoint_attributes(endpoint_arn='arn:myendpoint', attributes={'CustomUserData': 'john', 'Enabled': False})
    self.assert_request_parameters({'Action': 'SetEndpointAttributes', 'EndpointArn': 'arn:myendpoint', 'Attributes.entry.1.key': 'CustomUserData', 'Attributes.entry.1.value': 'john', 'Attributes.entry.2.key': 'Enabled', 'Attributes.entry.2.value': False}, ignore_params_values=['Version', 'ContentType'])