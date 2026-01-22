from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import boto
import os
import re
from gslib.commands import hmac
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
@SkipForS3('S3 does not have an equivalent API')
@SkipForXML('XML HMAC control is not supported.')
class TestHmacIntegration(testcase.GsUtilIntegrationTestCase):
    """Hmac integration test cases.

  These tests rely on the presence of 3 service accounts specified in the BOTO
  config. test_hmac_service_account and test_hmac_alt_service_account should not
  have any undeleted keys and test_hmac_list_service_account should have only
  deleted and active keys.
  """

    def ExtractAccessId(self, output_string):
        id_match = re.search('(GOOG[\\S]*)', output_string)
        if not id_match:
            self.fail('Couldn\'t find Access Id in output string:\n"%s"' % output_string)
        return id_match.group(0)

    def ExtractEtag(self, output_string):
        etag_match = re.search('\\sEtag:\\s+([\\S]+)$', output_string)
        if not etag_match:
            self.fail('Couldn\'t find Etag in output string:\n"%s"' % output_string)
        return etag_match.group(1)

    def AssertKeyMetadataMatches(self, output_string, access_id='GOOG.*', state='ACTIVE', service_account='.*', project='.*'):
        self.assertRegex(output_string, 'Access ID %s:' % access_id)
        self.assertRegex(output_string, '\\sState:\\s+%s' % state)
        self.assertRegex(output_string, '\\s+Service Account:\\s+%s\\n' % service_account)
        self.assertRegex(output_string, '\\s+Project:\\s+%s' % project)
        self.assertRegex(output_string, '\\s+Time Created:\\s+.*')
        self.assertRegex(output_string, '\\s+Time Last Updated:\\s+.*')

    def CleanupHelper(self, access_id):
        try:
            self.RunGsUtil(['hmac', 'update', '-s', 'INACTIVE', access_id])
        except AssertionError as e:
            if 'Update must modify the credential' not in str(e):
                raise
        self.RunGsUtil(['hmac', 'delete', access_id])

    @Retry(KeyLimitError, tries=5, timeout_secs=3)
    def _CreateWithRetry(self, service_account):
        """Retry creation on key limit failures."""
        try:
            return self.RunGsUtil(['hmac', 'create', service_account], return_stdout=True)
        except AssertionError as e:
            if 'HMAC key limit reached' in str(e):
                raise KeyLimitError(str(e))
            else:
                raise

    def CreateHelper(self, service_account):
        stdout = self._CreateWithRetry(service_account)
        return self.ExtractAccessId(stdout)

    def test_malformed_commands(self):
        params = [('hmac create', 'requires a service account', 'argument SERVICE_ACCOUNT: Must be specified'), ('hmac create -p proj', 'requires a service account', 'argument SERVICE_ACCOUNT: Must be specified'), ('hmac delete', 'requires an Access ID', 'argument ACCESS_ID: Must be specified'), ('hmac delete -p proj', 'requires an Access ID', 'argument ACCESS_ID: Must be specified'), ('hmac get', 'requires an Access ID', 'argument ACCESS_ID: Must be specified'), ('hmac get -p proj', 'requires an Access ID', 'argument ACCESS_ID: Must be specified'), ('hmac list account1', 'unexpected arguments', 'unrecognized arguments'), ('hmac update keyname', 'state flag must be supplied', 'Exactly one of (--activate | --deactivate) must be specified.'), ('hmac update -s INACTIVE', 'requires an Access ID', 'argument ACCESS_ID: Must be specified'), ('hmac update -s INACTIVE -p proj', 'requires an Access ID', 'argument ACCESS_ID: Must be specified')]
        for command, gsutil_error_substr, gcloud_error_substr in params:
            expected_status = 2 if self._use_gcloud_storage else 1
            stderr = self.RunGsUtil(command.split(), return_stderr=True, expected_status=expected_status)
            if self._use_gcloud_storage:
                self.assertIn(gcloud_error_substr, stderr)
            else:
                self.assertIn(gsutil_error_substr, stderr)

    def test_malformed_commands_that_cannot_be_translated_using_the_shim(self):
        if self._use_gcloud_storage:
            raise unittest.SkipTest('These commands cannot be translated using the shim.')
        params = [('hmac create -u email', 'requires a service account'), ('hmac update -s KENTUCKY', 'state flag value must be one of')]
        for command, gsutil_error_substr in params:
            stderr = self.RunGsUtil(command.split(), return_stderr=True, expected_status=1)
            self.assertIn(gsutil_error_substr, stderr)

    @unittest.skipUnless(SERVICE_ACCOUNT, 'Test requires service account configuration.')
    def test_create(self):
        stdout = self.RunGsUtil(['hmac', 'create', SERVICE_ACCOUNT], return_stdout=True)
        try:
            self.assertRegex(stdout, 'Access ID:\\s+\\S+')
            self.assertRegex(stdout, 'Secret:\\s+\\S+')
        finally:
            access_id = self.ExtractAccessId(stdout)
            self.CleanupHelper(access_id)

    def test_create_sa_not_found(self):
        stderr = self.RunGsUtil(['hmac', 'create', 'DNE@mail.com'], return_stderr=True, expected_status=1)
        self.assertIn("Service Account 'DNE@mail.com' not found.", stderr)

    @unittest.skipUnless(ALT_SERVICE_ACCOUNT, 'Test requires service account configuration.')
    def test_delete(self):
        access_id = self.CreateHelper(ALT_SERVICE_ACCOUNT)
        self.RunGsUtil(['hmac', 'update', '-s', 'INACTIVE', access_id])
        self.RunGsUtil(['hmac', 'delete', access_id])
        stdout = self.RunGsUtil(['hmac', 'list', '-u', ALT_SERVICE_ACCOUNT], return_stdout=True)
        self.assertNotIn(access_id, stdout)

    @unittest.skipUnless(SERVICE_ACCOUNT, 'Test requires service account configuration.')
    def test_delete_active_key(self):
        access_id = self.CreateHelper(SERVICE_ACCOUNT)
        stderr = self.RunGsUtil(['hmac', 'delete', access_id], return_stderr=True, expected_status=1)
        try:
            if self._use_gcloud_storage:
                self.assertIn('HTTPError 400: Cannot delete keys in ACTIVE state.', stderr)
            else:
                self.assertIn('400 Cannot delete keys in ACTIVE state.', stderr)
        finally:
            self.CleanupHelper(access_id)

    def test_delete_key_not_found(self):
        stderr = self.RunGsUtil(['hmac', 'delete', 'GOOG1234DNE'], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('HTTPError 404: Access ID not found in project', stderr)
        else:
            self.assertIn('404 Access ID not found', stderr)

    @unittest.skipUnless(ALT_SERVICE_ACCOUNT, 'Test requires service account configuration.')
    def test_get(self):
        access_id = self.CreateHelper(ALT_SERVICE_ACCOUNT)
        stdout = self.RunGsUtil(['hmac', 'get', access_id], return_stdout=True)
        try:
            self.AssertKeyMetadataMatches(stdout, access_id=access_id, service_account=ALT_SERVICE_ACCOUNT, project=PopulateProjectId(None))
        finally:
            self.CleanupHelper(access_id)

    def test_get_not_found(self):
        stderr = self.RunGsUtil(['hmac', 'get', 'GOOG1234DNE'], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('HTTPError 404: Access ID not found in project', stderr)
        else:
            self.assertIn('404 Access ID not found', stderr)

    def setUpListTest(self):
        for _ in range(MAX_SA_HMAC_KEYS):
            self.RunGsUtil(['hmac', 'create', LIST_SERVICE_ACCOUNT], expected_status=None)

    @unittest.skipUnless(LIST_SERVICE_ACCOUNT and SERVICE_ACCOUNT, 'Test requires service account configuration.')
    def test_list(self):
        self.setUpListTest()
        alt_access_id = self.CreateHelper(SERVICE_ACCOUNT)
        self.RunGsUtil(['hmac', 'update', '-s', 'INACTIVE', alt_access_id])
        stdout = self.RunGsUtil(['hmac', 'list', '-u', LIST_SERVICE_ACCOUNT], return_stdout=True)
        stdout = stdout.strip().split('\n')
        list_account_key_count = 0
        for line in stdout:
            access_id, state, account = line.split()
            self.assertIn('GOOG', access_id)
            self.assertEqual(account, LIST_SERVICE_ACCOUNT)
            self.assertEqual(state, 'ACTIVE')
            list_account_key_count += 1
        self.assertEqual(list_account_key_count, MAX_SA_HMAC_KEYS)
        stdout = self.RunGsUtil(['hmac', 'list'], return_stdout=True)
        stdout = stdout.strip().split('\n')
        project_key_count = 0
        inactive_key_listed = False
        for line in stdout:
            _, state, account = line.split()
            project_key_count += 1
            if account == SERVICE_ACCOUNT and state == 'INACTIVE':
                inactive_key_listed = True
        self.assertGreater(project_key_count, list_account_key_count)
        self.assertTrue(inactive_key_listed)
        self.RunGsUtil(['hmac', 'delete', alt_access_id])
        stdout = self.RunGsUtil(['hmac', 'list', '-a'], return_stdout=True)
        stdout = stdout.strip().split('\n')
        project_key_count = 0
        deleted_key_listed = False
        for line in stdout:
            _, state, account = line.split()
            project_key_count += 1
            if account == SERVICE_ACCOUNT and state == 'DELETED':
                deleted_key_listed = True
        self.assertTrue(deleted_key_listed)
        self.assertGreater(project_key_count, list_account_key_count)

    def ParseListOutput(self, stdout):
        current_key = ''
        for l in stdout.split('\n'):
            if current_key and l.startswith('Access ID'):
                yield current_key
                current_key = l
            else:
                current_key += l
            current_key += '\n'

    @unittest.skipUnless(LIST_SERVICE_ACCOUNT and ALT_SERVICE_ACCOUNT, 'Test requires service account configuration.')
    def test_list_long_format(self):
        self.setUpListTest()
        alt_access_id = self.CreateHelper(ALT_SERVICE_ACCOUNT)
        self.RunGsUtil(['hmac', 'update', '-s', 'INACTIVE', alt_access_id])
        stdout = self.RunGsUtil(['hmac', 'list', '-l'], return_stdout=True)
        try:
            self.assertIn(' ACTIVE', stdout)
            self.assertIn('INACTIVE', stdout)
            self.assertIn(ALT_SERVICE_ACCOUNT, stdout)
            self.assertIn(LIST_SERVICE_ACCOUNT, stdout)
            for key_metadata in self.ParseListOutput(stdout):
                self.AssertKeyMetadataMatches(key_metadata, state='.*')
        finally:
            self.CleanupHelper(alt_access_id)

    def test_list_service_account_not_found(self):
        service_account = 'service-account-DNE@gmail.com'
        stderr = self.RunGsUtil(['hmac', 'list', '-u', service_account], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn("HTTPError 404: Service Account '{}' not found.".format(service_account), stderr)
        else:
            self.assertIn("Service Account '%s' not found." % service_account, stderr)

    @unittest.skipUnless(ALT_SERVICE_ACCOUNT, 'Test requires service account configuration.')
    def test_update(self):
        access_id = self.CreateHelper(ALT_SERVICE_ACCOUNT)
        stdout = self.RunGsUtil(['hmac', 'get', access_id], return_stdout=True)
        etag = self.ExtractEtag(stdout)
        try:
            self.AssertKeyMetadataMatches(stdout, state='ACTIVE')
            stdout = self.RunGsUtil(['hmac', 'update', '-s', 'INACTIVE', '-e', etag, access_id], return_stdout=True)
            self.AssertKeyMetadataMatches(stdout, state='INACTIVE')
            stdout = self.RunGsUtil(['hmac', 'get', access_id], return_stdout=True)
            self.AssertKeyMetadataMatches(stdout, state='INACTIVE')
            stdout = self.RunGsUtil(['hmac', 'update', '-s', 'ACTIVE', access_id], return_stdout=True)
            self.AssertKeyMetadataMatches(stdout, state='ACTIVE')
            stdout = self.RunGsUtil(['hmac', 'get', access_id], return_stdout=True)
            self.AssertKeyMetadataMatches(stdout, state='ACTIVE')
            stderr = self.RunGsUtil(['hmac', 'update', '-s', 'INACTIVE', '-e', 'badEtag', access_id], return_stderr=True, expected_status=1)
            self.assertIn('Etag does not match expected value.', stderr)
        finally:
            self.CleanupHelper(access_id)