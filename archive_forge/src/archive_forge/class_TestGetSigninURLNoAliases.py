from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestGetSigninURLNoAliases(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n          <ListAccountAliasesResponse>\n            <ListAccountAliasesResult>\n              <IsTruncated>false</IsTruncated>\n              <AccountAliases></AccountAliases>\n            </ListAccountAliasesResult>\n            <ResponseMetadata>\n              <RequestId>c5a076e9-f1b0-11df-8fbe-45274EXAMPLE</RequestId>\n            </ResponseMetadata>\n          </ListAccountAliasesResponse>\n        '

    def test_get_signin_url_no_aliases(self):
        self.set_http_response(status_code=200)
        with self.assertRaises(Exception):
            self.service_connection.get_signin_url()