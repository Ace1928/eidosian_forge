from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestGetPolicy(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<GetPolicyResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <GetPolicyResult>\n    <Policy>\n      <PolicyName>S3-read-only-example-bucket</PolicyName>\n      <DefaultVersionId>v1</DefaultVersionId>\n      <PolicyId>AGPACKCEVSQ6C2EXAMPLE</PolicyId>\n      <Path>/</Path>\n      <Arn>arn:aws:iam::123456789012:policy/S3-read-only-example-bucket</Arn>\n      <AttachmentCount>9</AttachmentCount>\n      <CreateDate>2014-09-15T17:36:14Z</CreateDate>\n      <UpdateDate>2014-09-15T20:31:47Z</UpdateDate>\n      <Description>My Awesome Policy</Description>\n    </Policy>\n  </GetPolicyResult>\n  <ResponseMetadata>\n    <RequestId>684f0917-3d22-11e4-a4a0-cffb9EXAMPLE</RequestId>\n  </ResponseMetadata>\n</GetPolicyResponse>\n        '

    def test_get_policy(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_policy('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket')
        self.assert_request_parameters({'Action': 'GetPolicy', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket'}, ignore_params_values=['Version'])
        self.assertEqual(response['get_policy_response']['get_policy_result']['policy']['arn'], 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket')
        self.assertEqual(response['get_policy_response']['get_policy_result']['policy']['description'], 'My Awesome Policy')