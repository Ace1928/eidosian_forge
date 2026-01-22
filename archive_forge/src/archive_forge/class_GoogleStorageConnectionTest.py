import re
import sys
import copy
import json
import unittest
import email.utils
from io import BytesIO
from unittest import mock
from unittest.mock import Mock, PropertyMock
import pytest
from libcloud.test import StorageMockHttp
from libcloud.utils.py3 import StringIO, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_GOOGLE_STORAGE_PARAMS
from libcloud.common.google import GoogleAuthType
from libcloud.storage.drivers import google_storage
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
from libcloud.test.common.test_google import GoogleTestCase
class GoogleStorageConnectionTest(GoogleTestCase):

    @mock.patch('email.utils.formatdate')
    def test_add_default_headers(self, mock_formatdate):
        mock_formatdate.return_value = TODAY
        starting_headers = {'starting': 'headers'}
        project = 'foo-project'
        conn = CONN_CLS('foo_user', 'bar_key', secure=True, auth_type=GoogleAuthType.GCS_S3)
        conn.get_project = mock.Mock(return_value=None)
        headers = dict(starting_headers)
        headers['Date'] = TODAY
        self.assertEqual(conn.add_default_headers(dict(starting_headers)), headers)
        conn = CONN_CLS('foo_user', 'bar_key', secure=True, auth_type=GoogleAuthType.GCS_S3)
        conn.get_project = mock.Mock(return_value=project)
        headers = dict(starting_headers)
        headers['Date'] = TODAY
        headers[CONN_CLS.PROJECT_ID_HEADER] = project
        self.assertEqual(conn.add_default_headers(dict(starting_headers)), headers)

    @mock.patch('libcloud.storage.drivers.s3.BaseS3Connection.get_auth_signature')
    def test_get_s3_auth_signature(self, mock_s3_auth_sig_method):
        mock_s3_auth_sig_method.return_value = 'mock signature!'
        starting_params = {}
        starting_headers = {'Date': TODAY, 'x-goog-foo': 'X-GOOG: MAINTAIN UPPERCASE!', 'x-Goog-bar': 'Header key should be lowered', 'Content-Type': 'application/mIXED casING MAINTAINED', 'Other': 'LOWER THIS!'}
        modified_headers = {'date': TODAY, 'content-type': 'application/mIXED casING MAINTAINED', 'x-goog-foo': 'X-GOOG: MAINTAIN UPPERCASE!', 'x-goog-bar': 'Header key should be lowered', 'other': 'lower this!'}
        conn = CONN_CLS('foo_user', 'bar_key', secure=True, auth_type=GoogleAuthType.GCS_S3)
        conn.method = 'GET'
        conn.action = '/path'
        result = conn._get_s3_auth_signature(starting_params, starting_headers)
        self.assertNotEqual(starting_headers, modified_headers)
        self.assertEqual(result, 'mock signature!')
        mock_s3_auth_sig_method.assert_called_once_with(method='GET', headers=modified_headers, params=starting_params, expires=None, secret_key='bar_key', path='/path', vendor_prefix='x-goog')

    @mock.patch('libcloud.common.google.GoogleOAuth2Credential')
    def test_pre_connect_hook_oauth2(self, mock_oauth2_credential_init):
        mock_oauth2_credential_init.return_value = mock.Mock()
        starting_params = {'starting': 'params'}
        starting_headers = {'starting': 'headers'}
        conn = CONN_CLS('foo_user', 'bar_key', secure=True, auth_type=GoogleAuthType.GCE)
        conn._get_s3_auth_signature = mock.Mock()
        conn.oauth2_credential = mock.Mock()
        conn.oauth2_credential.access_token = 'Access_Token!'
        expected_headers = dict(starting_headers)
        expected_headers['Authorization'] = 'Bearer Access_Token!'
        result = conn.pre_connect_hook(dict(starting_params), dict(starting_headers))
        self.assertEqual(result, (starting_params, expected_headers))

    def test_pre_connect_hook_hmac(self):
        starting_params = {'starting': 'params'}
        starting_headers = {'starting': 'headers'}

        def fake_hmac_method(params, headers):
            fake_hmac_method.params_passed = copy.deepcopy(params)
            fake_hmac_method.headers_passed = copy.deepcopy(headers)
            return 'fake signature!'
        conn = CONN_CLS('foo_user', 'bar_key', secure=True, auth_type=GoogleAuthType.GCS_S3)
        conn._get_s3_auth_signature = fake_hmac_method
        conn.action = 'GET'
        conn.method = '/foo'
        expected_headers = dict(starting_headers)
        expected_headers['Authorization'] = '{} {}:{}'.format(google_storage.SIGNATURE_IDENTIFIER, 'foo_user', 'fake signature!')
        result = conn.pre_connect_hook(dict(starting_params), dict(starting_headers))
        self.assertEqual(result, (dict(starting_params), expected_headers))
        self.assertEqual(fake_hmac_method.params_passed, starting_params)
        self.assertEqual(fake_hmac_method.headers_passed, starting_headers)
        self.assertIsNone(conn.oauth2_credential)