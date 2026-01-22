from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import six
from gslib.commands import defacl
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as case
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
@SkipForS3('S3 does not support default object ACLs.')
class TestDefacl(case.GsUtilIntegrationTestCase):
    """Integration tests for the defacl command."""
    _defacl_ch_prefix = ['defacl', 'ch']
    _defacl_get_prefix = ['defacl', 'get']
    _defacl_set_prefix = ['defacl', 'set']

    def _MakeScopeRegex(self, role, entity_type, email_address):
        template_regex = '\\{.*"entity":\\s*"%s-%s".*"role":\\s*"%s".*\\}' % (entity_type, email_address, role)
        return re.compile(template_regex, flags=re.DOTALL)

    def testChangeDefaultAcl(self):
        """Tests defacl ch."""
        bucket = self.CreateBucket()
        test_regex = self._MakeScopeRegex('OWNER', 'group', self.GROUP_TEST_ADDRESS)
        test_regex2 = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._defacl_ch_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':FC', suri(bucket)])
        json_text2 = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        self.assertRegex(json_text2, test_regex)
        self.RunGsUtil(self._defacl_ch_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ', suri(bucket)])
        json_text3 = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        self.assertRegex(json_text3, test_regex2)
        stderr = self.RunGsUtil(self._defacl_ch_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':WRITE', suri(bucket)], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('WRITER is not a valid value', stderr)
        else:
            self.assertIn('WRITER cannot be set as a default object ACL', stderr)

    def testChangeDefaultAclEmpty(self):
        """Tests adding and removing an entry from an empty default object ACL."""
        bucket = self.CreateBucket()
        self.RunGsUtil(self._defacl_set_prefix + ['private', suri(bucket)])
        json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        empty_regex = '\\[\\]\\s*'
        self.assertRegex(json_text, empty_regex)
        group_regex = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
        self.RunGsUtil(self._defacl_ch_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ', suri(bucket)])
        json_text2 = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        self.assertRegex(json_text2, group_regex)
        if self.test_api == ApiSelector.JSON:
            return
        self.RunGsUtil(self._defacl_ch_prefix + ['-d', self.GROUP_TEST_ADDRESS, suri(bucket)])
        json_text3 = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        self.assertRegex(json_text3, empty_regex)

    def testChangeMultipleBuckets(self):
        """Tests defacl ch on multiple buckets."""
        bucket1 = self.CreateBucket()
        bucket2 = self.CreateBucket()
        test_regex = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket1)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket2)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._defacl_ch_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ', suri(bucket1), suri(bucket2)])
        json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket1)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket2)], return_stdout=True)
        self.assertRegex(json_text, test_regex)

    def testChangeMultipleAcls(self):
        """Tests defacl ch with multiple ACL entries."""
        bucket = self.CreateBucket()
        test_regex_group = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
        test_regex_user = self._MakeScopeRegex('OWNER', 'user', self.USER_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex_group)
        self.assertNotRegex(json_text, test_regex_user)
        self.RunGsUtil(self._defacl_ch_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ', '-u', self.USER_TEST_ADDRESS + ':fc', suri(bucket)])
        json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        self.assertRegex(json_text, test_regex_group)
        self.assertRegex(json_text, test_regex_user)

    def testEmptyDefAcl(self):
        bucket = self.CreateBucket()
        self.RunGsUtil(self._defacl_set_prefix + ['private', suri(bucket)])
        stdout = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        self.assertEqual(stdout.rstrip(), '[]')
        self.RunGsUtil(self._defacl_ch_prefix + ['-u', self.USER_TEST_ADDRESS + ':fc', suri(bucket)])

    def testDeletePermissionsWithCh(self):
        """Tests removing permissions with defacl ch."""
        bucket = self.CreateBucket()
        test_regex = self._MakeScopeRegex('OWNER', 'user', self.USER_TEST_ADDRESS)
        json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
        self.RunGsUtil(self._defacl_ch_prefix + ['-u', self.USER_TEST_ADDRESS + ':fc', suri(bucket)])
        json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
        self.RunGsUtil(self._defacl_ch_prefix + ['-d', self.USER_TEST_ADDRESS, suri(bucket)])
        json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)

    def testTooFewArgumentsFails(self):
        """Tests calling defacl with insufficient number of arguments."""
        stderr = self.RunGsUtil(self._defacl_get_prefix, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
        stderr = self.RunGsUtil(self._defacl_set_prefix, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
        stderr = self.RunGsUtil(self._defacl_ch_prefix, return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)
        stderr = self.RunGsUtil(['defacl'], return_stderr=True, expected_status=1)
        self.assertIn('command requires at least', stderr)