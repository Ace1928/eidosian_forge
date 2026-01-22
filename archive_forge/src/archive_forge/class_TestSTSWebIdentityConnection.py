from tests.unit import unittest
from boto.sts.connection import STSConnection
from tests.unit import AWSMockServiceTestCase
class TestSTSWebIdentityConnection(AWSMockServiceTestCase):
    connection_class = STSConnection

    def setUp(self):
        super(TestSTSWebIdentityConnection, self).setUp()

    def default_body(self):
        return b'\n<AssumeRoleWithWebIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">\n  <AssumeRoleWithWebIdentityResult>\n    <SubjectFromWebIdentityToken>\n      amzn1.account.AF6RHO7KZU5XRVQJGXK6HB56KR2A\n    </SubjectFromWebIdentityToken>\n    <AssumedRoleUser>\n      <Arn>\n        arn:aws:sts::000240903217:assumed-role/FederatedWebIdentityRole/app1\n      </Arn>\n      <AssumedRoleId>\n        AROACLKWSDQRAOFQC3IDI:app1\n      </AssumedRoleId>\n    </AssumedRoleUser>\n    <Credentials>\n      <SessionToken>\n        AQoDYXdzEE0a8ANXXXXXXXXNO1ewxE5TijQyp+IPfnyowF\n      </SessionToken>\n      <SecretAccessKey>\n        secretkey\n      </SecretAccessKey>\n      <Expiration>\n        2013-05-14T23:00:23Z\n      </Expiration>\n      <AccessKeyId>\n        accesskey\n      </AccessKeyId>\n    </Credentials>\n  </AssumeRoleWithWebIdentityResult>\n  <ResponseMetadata>\n    <RequestId>ad4156e9-bce1-11e2-82e6-6b6ef249e618</RequestId>\n  </ResponseMetadata>\n</AssumeRoleWithWebIdentityResponse>\n        '

    def test_assume_role_with_web_identity(self):
        arn = 'arn:aws:iam::000240903217:role/FederatedWebIdentityRole'
        wit = 'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        self.set_http_response(status_code=200)
        response = self.service_connection.assume_role_with_web_identity(role_arn=arn, role_session_name='guestuser', web_identity_token=wit, provider_id='www.amazon.com')
        self.assert_request_parameters({'RoleSessionName': 'guestuser', 'RoleArn': arn, 'WebIdentityToken': wit, 'ProviderId': 'www.amazon.com', 'Action': 'AssumeRoleWithWebIdentity'}, ignore_params_values=['Version'])
        self.assertEqual(response.credentials.access_key.strip(), 'accesskey')
        self.assertEqual(response.credentials.secret_key.strip(), 'secretkey')
        self.assertEqual(response.credentials.session_token.strip(), 'AQoDYXdzEE0a8ANXXXXXXXXNO1ewxE5TijQyp+IPfnyowF')
        self.assertEqual(response.user.arn.strip(), 'arn:aws:sts::000240903217:assumed-role/FederatedWebIdentityRole/app1')
        self.assertEqual(response.user.assume_role_id.strip(), 'AROACLKWSDQRAOFQC3IDI:app1')