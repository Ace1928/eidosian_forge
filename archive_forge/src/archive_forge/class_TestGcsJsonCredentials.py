from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import GceAssertionCredentials
from google_reauth import reauth_creds
from gslib import gcs_json_api
from gslib import gcs_json_credentials
from gslib.cred_types import CredTypes
from gslib.exception import CommandException
from gslib.tests import testcase
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.utils.wrapped_credentials import WrappedCredentials
import logging
from oauth2client.service_account import ServiceAccountCredentials
import pkgutil
from six import add_move, MovedModule
from six.moves import mock
class TestGcsJsonCredentials(testcase.GsUtilUnitTestCase):
    """Test logic for interacting with GCS JSON Credentials."""

    @unittest.skipUnless(HAS_OPENSSL, 'signurl requires pyopenssl.')
    def testOauth2ServiceAccountCredential(self):
        contents = pkgutil.get_data('gslib', 'tests/test_data/test.p12')
        tmpfile = self.CreateTempFile(contents=contents)
        with SetBotoConfigForTest(getBotoCredentialsConfig(service_account_creds={'keyfile': tmpfile, 'client_id': '?'})):
            self.assertTrue(gcs_json_credentials._HasOauth2ServiceAccountCreds())
            client = gcs_json_api.GcsJsonApi(None, None, None, None)
            self.assertIsInstance(client.credentials, ServiceAccountCredentials)

    @unittest.skipUnless(HAS_OPENSSL, 'signurl requires pyopenssl.')
    @mock.patch.object(ServiceAccountCredentials, '__init__', side_effect=ValueError(ERROR_MESSAGE))
    def testOauth2ServiceAccountFailure(self, _):
        contents = pkgutil.get_data('gslib', 'tests/test_data/test.p12')
        tmpfile = self.CreateTempFile(contents=contents)
        with SetBotoConfigForTest(getBotoCredentialsConfig(service_account_creds={'keyfile': tmpfile, 'client_id': '?'})):
            with self.assertLogs() as logger:
                with self.assertRaises(Exception) as exc:
                    gcs_json_api.GcsJsonApi(None, logging.getLogger(), None, None)
                self.assertIn(ERROR_MESSAGE, str(exc.exception))
                self.assertIn(CredTypes.OAUTH2_SERVICE_ACCOUNT, logger.output[0])

    def testOauth2UserCredential(self):
        with SetBotoConfigForTest(getBotoCredentialsConfig(user_account_creds='?')):
            self.assertTrue(gcs_json_credentials._HasOauth2UserAccountCreds())
            client = gcs_json_api.GcsJsonApi(None, None, None, None)
            self.assertIsInstance(client.credentials, reauth_creds.Oauth2WithReauthCredentials)

    @mock.patch.object(reauth_creds.Oauth2WithReauthCredentials, '__init__', side_effect=ValueError(ERROR_MESSAGE))
    def testOauth2UserFailure(self, _):
        with SetBotoConfigForTest(getBotoCredentialsConfig(user_account_creds='?')):
            with self.assertLogs() as logger:
                with self.assertRaises(Exception) as exc:
                    gcs_json_api.GcsJsonApi(None, logging.getLogger(), None, None)
                self.assertIn(ERROR_MESSAGE, str(exc.exception))
                self.assertIn(CredTypes.OAUTH2_USER_ACCOUNT, logger.output[0])

    @mock.patch.object(gcs_json_credentials.credentials_lib, 'GceAssertionCredentials', autospec=True)
    def testGCECredential(self, mock_credentials):

        def set_store(store):
            mock_credentials.return_value.store = store
        mock_credentials.return_value.client_id = None
        mock_credentials.return_value.refresh_token = 'rEfrEshtOkEn'
        mock_credentials.return_value.set_store = set_store
        with SetBotoConfigForTest(getBotoCredentialsConfig(gce_creds='?')):
            self.assertTrue(gcs_json_credentials._HasGceCreds())
            client = gcs_json_api.GcsJsonApi(None, None, None, None)
            self.assertIsInstance(client.credentials, GceAssertionCredentials)
            self.assertEqual(client.credentials.refresh_token, 'rEfrEshtOkEn')
            self.assertIs(client.credentials.client_id, None)

    @mock.patch.object(GceAssertionCredentials, '__init__', side_effect=ValueError(ERROR_MESSAGE))
    def testGCECredentialFailure(self, _):
        with SetBotoConfigForTest(getBotoCredentialsConfig(gce_creds='?')):
            with self.assertLogs() as logger:
                with self.assertRaises(Exception) as exc:
                    gcs_json_api.GcsJsonApi(None, logging.getLogger(), None, None)
                self.assertIn(ERROR_MESSAGE, str(exc.exception))
                self.assertIn(CredTypes.GCE, logger.output[0])

    def testExternalAccountCredential(self):
        contents = pkgutil.get_data('gslib', 'tests/test_data/test_external_account_credentials.json')
        tmpfile = self.CreateTempFile(contents=contents)
        with SetBotoConfigForTest(getBotoCredentialsConfig(external_account_creds=tmpfile)):
            client = gcs_json_api.GcsJsonApi(None, None, None, None)
            self.assertIsInstance(client.credentials, WrappedCredentials)

    @mock.patch.object(WrappedCredentials, '__init__', side_effect=ValueError(ERROR_MESSAGE))
    def testExternalAccountFailure(self, _):
        contents = pkgutil.get_data('gslib', 'tests/test_data/test_external_account_credentials.json')
        tmpfile = self.CreateTempFile(contents=contents)
        with SetBotoConfigForTest(getBotoCredentialsConfig(external_account_creds=tmpfile)):
            with self.assertLogs() as logger:
                with self.assertRaises(Exception) as exc:
                    gcs_json_api.GcsJsonApi(None, logging.getLogger(), None, None)
                self.assertIn(ERROR_MESSAGE, str(exc.exception))
                self.assertIn(CredTypes.EXTERNAL_ACCOUNT, logger.output[0])

    def testExternalAccountAuthorizedUserCredential(self):
        contents = pkgutil.get_data('gslib', 'tests/test_data/test_external_account_authorized_user_credentials.json')
        tmpfile = self.CreateTempFile(contents=contents)
        with SetBotoConfigForTest(getBotoCredentialsConfig(external_account_authorized_user_creds=tmpfile)):
            client = gcs_json_api.GcsJsonApi(None, None, None, None)
            self.assertIsInstance(client.credentials, WrappedCredentials)

    @mock.patch.object(WrappedCredentials, '__init__', side_effect=ValueError(ERROR_MESSAGE))
    def testExternalAccountAuthorizedUserFailure(self, _):
        contents = pkgutil.get_data('gslib', 'tests/test_data/test_external_account_authorized_user_credentials.json')
        tmpfile = self.CreateTempFile(contents=contents)
        with SetBotoConfigForTest(getBotoCredentialsConfig(external_account_authorized_user_creds=tmpfile)):
            with self.assertLogs() as logger:
                with self.assertRaises(Exception) as exc:
                    gcs_json_api.GcsJsonApi(None, logging.getLogger(), None, None)
                self.assertIn(ERROR_MESSAGE, str(exc.exception))
                self.assertIn(CredTypes.EXTERNAL_ACCOUNT_AUTHORIZED_USER, logger.output[0])

    def testOauth2ServiceAccountAndOauth2UserCredential(self):
        with SetBotoConfigForTest(getBotoCredentialsConfig(user_account_creds='?', service_account_creds={'keyfile': '?', 'client_id': '?'})):
            with self.assertRaises(CommandException):
                gcs_json_api.GcsJsonApi(None, None, None, None)