from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestGetPolicyVersion(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<GetPolicyVersionResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <GetPolicyVersionResult>\n    <PolicyVersion>\n      <Document>\n      {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:Get*","s3:List*"],\n      "Resource":["arn:aws:s3:::EXAMPLE-BUCKET","arn:aws:s3:::EXAMPLE-BUCKET/*"]}]}\n      </Document>\n      <IsDefaultVersion>true</IsDefaultVersion>\n      <VersionId>v1</VersionId>\n      <CreateDate>2014-09-15T20:31:47Z</CreateDate>\n    </PolicyVersion>\n  </GetPolicyVersionResult>\n  <ResponseMetadata>\n    <RequestId>d472f28e-3d23-11e4-a4a0-cffb9EXAMPLE</RequestId>\n  </ResponseMetadata>\n</GetPolicyVersionResponse>\n        '

    def test_get_policy_version(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_policy_version('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'v1')
        self.assert_request_parameters({'Action': 'GetPolicyVersion', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'VersionId': 'v1'}, ignore_params_values=['Version'])
        self.assertEqual(response['get_policy_version_response']['get_policy_version_result']['policy_version']['version_id'], 'v1')