from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestGetSigninURL(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n          <ListAccountAliasesResponse>\n            <ListAccountAliasesResult>\n              <IsTruncated>false</IsTruncated>\n              <AccountAliases>\n                <member>foocorporation</member>\n                <member>anotherunused</member>\n              </AccountAliases>\n            </ListAccountAliasesResult>\n            <ResponseMetadata>\n              <RequestId>c5a076e9-f1b0-11df-8fbe-45274EXAMPLE</RequestId>\n            </ResponseMetadata>\n          </ListAccountAliasesResponse>\n        '

    def test_get_signin_url_default(self):
        self.set_http_response(status_code=200)
        url = self.service_connection.get_signin_url()
        self.assertEqual(url, 'https://foocorporation.signin.aws.amazon.com/console/ec2')

    def test_get_signin_url_s3(self):
        self.set_http_response(status_code=200)
        url = self.service_connection.get_signin_url(service='s3')
        self.assertEqual(url, 'https://foocorporation.signin.aws.amazon.com/console/s3')

    def test_get_signin_url_cn_north(self):
        self.set_http_response(status_code=200)
        self.service_connection.host = 'iam.cn-north-1.amazonaws.com.cn'
        url = self.service_connection.get_signin_url()
        self.assertEqual(url, 'https://foocorporation.signin.amazonaws.cn/console/ec2')