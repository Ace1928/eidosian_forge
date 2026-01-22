from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestAttachGroupPolicy(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<AttachGroupPolicyResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <ResponseMetadata>\n    <RequestId>f8a7b7b9-3d01-11e4-bfad-8d1c6EXAMPLE</RequestId>\n  </ResponseMetadata>\n</AttachGroupPolicyResponse>\n        '

    def test_attach_group_policy(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.attach_group_policy('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'Dev')
        self.assert_request_parameters({'Action': 'AttachGroupPolicy', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'GroupName': 'Dev'}, ignore_params_values=['Version'])
        self.assertEqual('request_id' in response['attach_group_policy_response']['response_metadata'], True)