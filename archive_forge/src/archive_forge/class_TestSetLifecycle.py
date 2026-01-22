from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import os
import posixpath
from unittest import mock
from xml.dom.minidom import parseString
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils.translation_helper import LifecycleTranslation
from gslib.utils import shim_util
@SkipForS3('Lifecycle command is only supported for gs:// URLs')
class TestSetLifecycle(testcase.GsUtilIntegrationTestCase):
    """Integration tests for lifecycle command."""
    empty_doc1 = '{}'
    xml_doc = parseString('<LifecycleConfiguration><Rule><Action><Delete/></Action><Condition><Age>365</Age></Condition></Rule></LifecycleConfiguration>').toprettyxml(indent='    ')
    bad_doc = '{"rule": [{"action": {"type": "Add"}, "condition": {"age": 365}}]}\n'
    lifecycle_doc = '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 365}}, {"action": {"type": "SetStorageClass", "storageClass": "NEARLINE"}, "condition": {"matchesStorageClass": ["STANDARD"], "age": 366}}]}\n'
    lifecycle_json_obj = json.loads(lifecycle_doc)
    lifecycle_doc_bucket_style = '{"lifecycle": ' + lifecycle_doc.rstrip() + '}\n'
    lifecycle_doc_without_storage_class_fields = '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 365}}]}\n'
    lifecycle_created_before_doc = '{"rule": [{"action": {"type": "Delete"}, "condition": {"createdBefore": "2014-10-01"}}]}\n'
    lifecycle_created_before_json_obj = json.loads(lifecycle_created_before_doc)
    lifecycle_with_falsy_condition_values = '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 0, "isLive": false, "numNewerVersions": 0}}]}'
    no_lifecycle_config = 'has no lifecycle configuration.'
    empty_lifecycle_config = '[]'

    def test_lifecycle_translation(self):
        """Tests lifecycle translation for various formats."""
        json_text = self.lifecycle_doc_without_storage_class_fields
        entries_list = LifecycleTranslation.JsonLifecycleToMessage(json_text)
        boto_lifecycle = LifecycleTranslation.BotoLifecycleFromMessage(entries_list)
        converted_entries_list = LifecycleTranslation.BotoLifecycleToMessage(boto_lifecycle)
        converted_json_text = LifecycleTranslation.JsonLifecycleFromMessage(converted_entries_list)
        self.assertEqual(json.loads(json_text), json.loads(converted_json_text))

    def test_default_lifecycle(self):
        bucket_uri = self.CreateBucket()
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket_uri)], return_stdout=True)
        if self._use_gcloud_storage:
            self.assertIn(self.empty_lifecycle_config, stdout)
        else:
            self.assertIn(self.no_lifecycle_config, stdout)

    def test_set_empty_lifecycle1(self):
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=self.empty_doc1.encode('ascii'))
        self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)])
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket_uri)], return_stdout=True)
        if self._use_gcloud_storage:
            self.assertIn(self.empty_lifecycle_config, stdout)
        else:
            self.assertIn(self.no_lifecycle_config, stdout)

    def test_valid_lifecycle(self):
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=self.lifecycle_doc.encode('ascii'))
        self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)])
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket_uri)], return_stdout=True)
        self.assertEqual(json.loads(stdout), self.lifecycle_json_obj)

    def test_valid_lifecycle_bucket_style(self):
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=self.lifecycle_doc_bucket_style.encode('ascii'))
        self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)])
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket_uri)], return_stdout=True)
        self.assertEqual(json.loads(stdout), self.lifecycle_json_obj)

    def test_created_before_lifecycle(self):
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=self.lifecycle_created_before_doc.encode('ascii'))
        self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)])
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket_uri)], return_stdout=True)
        self.assertEqual(json.loads(stdout), self.lifecycle_created_before_json_obj)

    def test_bad_lifecycle(self):
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=self.bad_doc.encode('ascii'))
        stderr = self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
        self.assertNotIn('XML lifecycle data provided', stderr)

    def test_bad_xml_lifecycle(self):
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=self.xml_doc.encode('ascii'))
        stderr = self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
        self.assertIn('XML lifecycle data provided', stderr)

    def test_translation_for_falsy_values_works_correctly(self):
        bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=self.lifecycle_with_falsy_condition_values.encode('ascii'))
        self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)])
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket_uri)], return_stdout=True)
        self.assertRegex(stdout, '"age":\\s+0')
        self.assertRegex(stdout, '"isLive":\\s+false')
        self.assertRegex(stdout, '"numNewerVersions":\\s+0')

    def test_set_lifecycle_and_reset(self):
        """Tests setting and turning off lifecycle configuration."""
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        fpath = self.CreateTempFile(tmpdir=tmpdir, contents=self.lifecycle_doc.encode('ascii'))
        self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)])
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket_uri)], return_stdout=True)
        self.assertEqual(json.loads(stdout), self.lifecycle_json_obj)
        fpath = self.CreateTempFile(tmpdir=tmpdir, contents=self.empty_doc1.encode('ascii'))
        self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)])
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket_uri)], return_stdout=True)
        if self._use_gcloud_storage:
            self.assertIn(self.empty_lifecycle_config, stdout)
        else:
            self.assertIn(self.no_lifecycle_config, stdout)

    def test_set_lifecycle_multi_buckets(self):
        """Tests setting lifecycle configuration on multiple buckets."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        fpath = self.CreateTempFile(contents=self.lifecycle_doc.encode('ascii'))
        self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket1_uri), suri(bucket2_uri)])
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket1_uri)], return_stdout=True)
        self.assertEqual(json.loads(stdout), self.lifecycle_json_obj)
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket2_uri)], return_stdout=True)
        self.assertEqual(json.loads(stdout), self.lifecycle_json_obj)

    def test_set_lifecycle_wildcard(self):
        """Tests setting lifecycle with a wildcarded bucket URI."""
        if self.test_api == ApiSelector.XML:
            return unittest.skip('XML wildcard behavior can cause test to flake if a bucket in the same project is deleted during execution.')
        random_prefix = self.MakeRandomTestString()
        bucket1_name = self.MakeTempName('bucket', prefix=random_prefix)
        bucket2_name = self.MakeTempName('bucket', prefix=random_prefix)
        bucket1_uri = self.CreateBucket(bucket_name=bucket1_name)
        bucket2_uri = self.CreateBucket(bucket_name=bucket2_name)
        common_prefix = posixpath.commonprefix([suri(bucket1_uri), suri(bucket2_uri)])
        self.assertTrue(common_prefix.startswith('gs://%sgsutil-test-test-set-lifecycle-wildcard-' % random_prefix))
        wildcard = '%s*' % common_prefix
        fpath = self.CreateTempFile(contents=self.lifecycle_doc.encode('ascii'))
        actual_lines = set()

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _Check1():
            stderr = self.RunGsUtil(['lifecycle', 'set', fpath, wildcard], return_stderr=True)
            actual_lines.update(stderr.splitlines())
            if self._use_gcloud_storage:
                self.assertIn('Updating %s/...' % suri(bucket1_uri), stderr)
                self.assertIn('Updating %s/...' % suri(bucket2_uri), stderr)
                status_message = 'Updating'
            else:
                expected_lines = set(['Setting lifecycle configuration on %s/...' % suri(bucket1_uri), 'Setting lifecycle configuration on %s/...' % suri(bucket2_uri)])
                self.assertEqual(expected_lines, actual_lines)
                status_message = 'Setting lifecycle configuration'
            self.assertEqual(stderr.count(status_message), 2)
        _Check1()
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket1_uri)], return_stdout=True)
        self.assertEqual(json.loads(stdout), self.lifecycle_json_obj)
        stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket2_uri)], return_stdout=True)
        self.assertEqual(json.loads(stdout), self.lifecycle_json_obj)