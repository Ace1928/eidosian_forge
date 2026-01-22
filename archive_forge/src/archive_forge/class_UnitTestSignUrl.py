from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from datetime import datetime
from datetime import timedelta
import os
import pkgutil
import boto
import gslib.commands.signurl
from gslib.commands.signurl import HAVE_OPENSSL
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.iamcredentials_api import IamcredentailsApi
from gslib.impersonation_credentials import ImpersonationCredentials
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import (SkipForS3, SkipForXML)
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
import gslib.tests.signurl_signatures as sigs
from oauth2client import client
from oauth2client.service_account import ServiceAccountCredentials
from six import add_move, MovedModule
from six.moves import mock
@unittest.skipUnless(HAVE_OPENSSL, 'signurl requires pyopenssl.')
class UnitTestSignUrl(testcase.GsUtilUnitTestCase):
    """Unit tests for the signurl command."""
    maxDiff = None

    def setUp(self):
        super(UnitTestSignUrl, self).setUp()
        ks_contents = pkgutil.get_data('gslib', 'tests/test_data/test.p12')
        self.key, self.client_email = gslib.commands.signurl._ReadKeystore(ks_contents, 'notasecret')

        def fake_now():
            return datetime(1900, 1, 1, 0, 5, 55)
        gslib.utils.signurl_helper._NowUTC = fake_now

    def _get_mock_api_delegator(self):
        mock_api_delegator = self.MakeGsUtilApi()
        mock_api_delegator.api_map['apiclass']['gs']['JSON'] = GcsJsonApi
        return mock_api_delegator

    def testDurationSpec(self):
        tests = [('1h', timedelta(hours=1)), ('2d', timedelta(days=2)), ('5D', timedelta(days=5)), ('35s', timedelta(seconds=35)), ('1h', timedelta(hours=1)), ('33', timedelta(hours=33)), ('22m', timedelta(minutes=22)), ('3.7', None), ('27Z', None)]
        for inp, expected in tests:
            try:
                td = gslib.commands.signurl._DurationToTimeDelta(inp)
                self.assertEqual(td, expected)
            except CommandException:
                if expected is not None:
                    self.fail('{0} failed to parse')

    def testSignPutUsingKeyFile(self):
        """Tests the _GenSignedUrl function with a PUT method using Key file."""
        expected = sigs.TEST_SIGN_PUT_SIG
        duration = timedelta(seconds=3600)
        with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
            signed_url = gslib.commands.signurl._GenSignedUrl(self.key, api=None, use_service_account=False, provider='gs', client_id=self.client_email, method='RESUMABLE', gcs_path='test/test.txt', duration=duration, logger=self.logger, region='us-east', content_type='')
        self.assertEqual(expected, signed_url)

    @SkipForS3('Tests only uses gs credentials.')
    @SkipForXML('Tests only run on JSON API.')
    def testSignPutUsingServiceAccount(self):
        """Tests the _GenSignedUrl function PUT method with service account."""
        expected = sigs.TEST_SIGN_URL_PUT_WITH_SERVICE_ACCOUNT
        duration = timedelta(seconds=3600)
        mock_api_delegator = self._get_mock_api_delegator()
        json_api = mock_api_delegator._GetApi('gs')
        mock_credentials = mock.Mock(spec=ServiceAccountCredentials)
        mock_credentials.service_account_email = 'fake_service_account_email'
        mock_credentials.sign_blob.return_value = ('fake_key', b'fake_signature')
        json_api.credentials = mock_credentials
        with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
            signed_url = gslib.commands.signurl._GenSignedUrl(None, api=mock_api_delegator, use_service_account=True, provider='gs', client_id=self.client_email, method='PUT', gcs_path='test/test.txt', duration=duration, logger=self.logger, region='us-east1', content_type='')
        self.assertEqual(expected, signed_url)
        mock_credentials.sign_blob.assert_called_once_with(b'GOOG4-RSA-SHA256\n19000101T000555Z\n19000101/us-east1/storage/goog4_request\n7f110b30eeca7fdd8846e876bceee85384d8e4c7388b3596544b1b503f9e2320')

    @SkipForS3('Tests only uses gs credentials.')
    @SkipForXML('Tests only run on JSON API.')
    def testSignUrlWithIncorrectAccountType(self):
        """Tests the _GenSignedUrl with incorrect account type.

    Test that GenSignedUrl function with 'use_service_account' set to True
    and a service account not used for credentials raises an error.
    """
        expected = sigs.TEST_SIGN_URL_PUT_WITH_SERVICE_ACCOUNT
        duration = timedelta(seconds=3600)
        mock_api_delegator = self._get_mock_api_delegator()
        json_api = mock_api_delegator._GetApi('gs')
        mock_credentials = mock.Mock(spec=client.OAuth2Credentials)
        mock_credentials.service_account_email = 'fake_service_account_email'
        json_api.credentials = mock_credentials
        with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
            self.assertRaises(CommandException, gslib.commands.signurl._GenSignedUrl, None, api=mock_api_delegator, use_service_account=True, provider='gs', client_id=self.client_email, method='PUT', gcs_path='test/test.txt', duration=duration, logger=self.logger, region='us-east1', content_type='')

    @SkipForS3('Tests only uses gs credentials.')
    @SkipForXML('Tests only run on JSON API.')
    @mock.patch('gslib.iamcredentials_api.apitools_client')
    @mock.patch('gslib.iamcredentials_api.apitools_messages')
    def testSignPutUsingImersonatedServiceAccount(self, mock_api_messages, mock_apiclient):
        """Tests the _GenSignedUrl function PUT method with impersonation.

    Test _GenSignedUrl function using an impersonated service account.
    """
        expected = sigs.TEST_SIGN_URL_PUT_WITH_SERVICE_ACCOUNT
        duration = timedelta(seconds=3600)
        mock_api_delegator = self._get_mock_api_delegator()
        json_api = mock_api_delegator._GetApi('gs')
        mock_credentials = mock.Mock(spec=ImpersonationCredentials)
        api_client_obj = mock.Mock()
        mock_apiclient.IamcredentialsV1.return_value = api_client_obj
        mock_iam_cred_api = IamcredentailsApi(credentials=mock.Mock())
        mock_credentials.api = mock_iam_cred_api
        mock_credentials.service_account_id = 'fake_service_account_email'
        mock_resp = mock.Mock()
        mock_resp.signedBlob = b'fake_signature'
        api_client_obj.projects_serviceAccounts.SignBlob.return_value = mock_resp
        json_api.credentials = mock_credentials
        with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
            signed_url = gslib.commands.signurl._GenSignedUrl(None, api=mock_api_delegator, use_service_account=True, provider='gs', client_id=self.client_email, method='PUT', gcs_path='test/test.txt', duration=duration, logger=self.logger, region='us-east1', content_type='')
        self.assertEqual(expected, signed_url)
        mock_api_messages.SignBlobRequest.assert_called_once_with(payload=b'GOOG4-RSA-SHA256\n19000101T000555Z\n19000101/us-east1/storage/goog4_request\n7f110b30eeca7fdd8846e876bceee85384d8e4c7388b3596544b1b503f9e2320')

    def testSignResumableWithKeyFile(self):
        """Tests _GenSignedUrl using key file with a RESUMABLE method."""
        expected = sigs.TEST_SIGN_RESUMABLE

        class MockLogger(object):

            def __init__(self):
                self.warning_issued = False

            def warn(self, unused_msg):
                self.warning_issued = True
        mock_logger = MockLogger()
        duration = timedelta(seconds=3600)
        with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
            signed_url = gslib.commands.signurl._GenSignedUrl(self.key, api=None, use_service_account=False, provider='gs', client_id=self.client_email, method='RESUMABLE', gcs_path='test/test.txt', duration=duration, logger=mock_logger, region='us-east', content_type='')
        self.assertEqual(expected, signed_url)
        self.assertTrue(mock_logger.warning_issued)
        mock_logger2 = MockLogger()
        with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
            signed_url = gslib.commands.signurl._GenSignedUrl(self.key, api=None, use_service_account=False, provider='gs', client_id=self.client_email, method='RESUMABLE', gcs_path='test/test.txt', duration=duration, logger=mock_logger2, region='us-east', content_type='image/jpeg')
        self.assertFalse(mock_logger2.warning_issued)

    def testSignurlPutContentypeUsingKeyFile(self):
        """Tests _GenSignedUrl using key file with a PUT method and content type."""
        expected = sigs.TEST_SIGN_URL_PUT_CONTENT
        duration = timedelta(seconds=3600)
        with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
            signed_url = gslib.commands.signurl._GenSignedUrl(self.key, api=None, use_service_account=False, provider='gs', client_id=self.client_email, method='PUT', gcs_path='test/test.txt', duration=duration, logger=self.logger, region='eu', content_type='text/plain')
        self.assertEqual(expected, signed_url)

    def testSignurlGetUsingKeyFile(self):
        """Tests the _GenSignedUrl function using key file with a GET method."""
        expected = sigs.TEST_SIGN_URL_GET
        duration = timedelta(seconds=0)
        with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
            signed_url = gslib.commands.signurl._GenSignedUrl(self.key, api=None, use_service_account=False, provider='gs', client_id=self.client_email, method='GET', gcs_path='test/test.txt', duration=duration, logger=self.logger, region='asia', content_type='')
        self.assertEqual(expected, signed_url)

    def testSignurlGetWithJSONKeyUsingKeyFile(self):
        """Tests _GenSignedUrl with a GET method and the test JSON private key."""
        expected = sigs.TEST_SIGN_URL_GET_WITH_JSON_KEY
        json_contents = pkgutil.get_data('gslib', 'tests/test_data/test.json').decode()
        key, client_email = gslib.commands.signurl._ReadJSONKeystore(json_contents)
        duration = timedelta(seconds=0)
        with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
            signed_url = gslib.commands.signurl._GenSignedUrl(key, api=None, use_service_account=False, provider='gs', client_id=client_email, method='GET', gcs_path='test/test.txt', duration=duration, logger=self.logger, region='asia', content_type='')
        self.assertEqual(expected, signed_url)

    def testSignurlGetWithUserProject(self):
        """Tests the _GenSignedUrl function with a userproject."""
        expected = sigs.TEST_SIGN_URL_GET_USERPROJECT
        duration = timedelta(seconds=0)
        with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
            signed_url = gslib.commands.signurl._GenSignedUrl(self.key, api=None, use_service_account=False, provider='gs', client_id=self.client_email, method='GET', gcs_path='test/test.txt', duration=duration, logger=self.logger, region='asia', content_type='', billing_project='myproject')
        self.assertEqual(expected, signed_url)

    def testShimTranslatesFlags(self):
        key_contents = pkgutil.get_data('gslib', 'tests/test_data/test.json')
        key_path = self.CreateTempFile(contents=key_contents)
        with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'dry_run')]):
            with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
                mock_log_handler = self.RunCommand('signurl', ['-d', '2m', '-m', 'RESUMABLE', '-r', 'US', '-b', 'project', '-c', 'application/octet-stream', key_path, 'gs://bucket/object'], return_log_handler=True)
                info_lines = '\n'.join(mock_log_handler.messages['info'])
                self.assertIn('storage sign-url --format=csv[separator="\\t"](resource:label="URL", http_verb:label="HTTP Method", expiration:label="Expiration", signed_url:label="Signed URL") --private-key-file={} --headers=x-goog-resumable=start --duration 120s --http-verb POST --region US --query-params userProject=project --headers content-type=application/octet-stream gs://bucket/object'.format(key_path), info_lines)