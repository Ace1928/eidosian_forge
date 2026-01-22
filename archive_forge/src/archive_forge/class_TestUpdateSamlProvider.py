from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestUpdateSamlProvider(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n            <UpdateSAMLProviderResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n              <UpdateSAMLProviderResult>\n                <SAMLProviderArn>arn:aws:iam::123456789012:saml-metadata/MyUniversity</SAMLProviderArn>\n              </UpdateSAMLProviderResult>\n              <ResponseMetadata>\n                <RequestId>29f47818-99f5-11e1-a4c3-27EXAMPLE804</RequestId>\n              </ResponseMetadata>\n            </UpdateSAMLProviderResponse>\n        '

    def test_update_saml_provider(self):
        self.set_http_response(status_code=200)
        self.service_connection.update_saml_provider('arn', 'doc')
        self.assert_request_parameters({'Action': 'UpdateSAMLProvider', 'SAMLMetadataDocument': 'doc', 'SAMLProviderArn': 'arn'}, ignore_params_values=['Version'])