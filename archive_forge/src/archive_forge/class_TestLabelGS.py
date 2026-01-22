from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import xml
from xml.dom.minidom import parseString
from xml.sax import _exceptions as SaxExceptions
import six
import boto
from boto import handler
from boto.s3.tagging import Tags
from gslib.exception import CommandException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retry_util import Retry
from gslib.utils.constants import UTF8
@SkipForS3('Tests use GS-style ')
class TestLabelGS(testcase.GsUtilIntegrationTestCase):
    """Integration tests for label command."""
    _label_dict = {KEY1: VALUE1, KEY2: VALUE2}

    def setUp(self):
        super(TestLabelGS, self).setUp()
        self.json_fpath = self.CreateTempFile(contents=json.dumps(self._label_dict).encode(UTF8))

    def testSetAndGetOnOneBucket(self):
        bucket_uri = self.CreateBucket()
        stderr = self.RunGsUtil(['label', 'set', self.json_fpath, suri(bucket_uri)], return_stderr=True)
        expected_output = _get_label_setting_output(self._use_gcloud_storage, suri(bucket_uri))
        if self._use_gcloud_storage:
            self.assertIn(expected_output, stderr)
        else:
            self.assertEqual(stderr.strip(), expected_output)
        stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
        self.assertDictEqual(json.loads(stdout), self._label_dict)

    def testSetOnMultipleBucketsInSameCommand(self):
        bucket_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        stderr = self.RunGsUtil(['label', 'set', self.json_fpath, suri(bucket_uri), suri(bucket2_uri)], return_stderr=True)
        actual = set(stderr.splitlines())
        expected = set([_get_label_setting_output(self._use_gcloud_storage, suri(bucket_uri)), _get_label_setting_output(self._use_gcloud_storage, suri(bucket2_uri))])
        if self._use_gcloud_storage:
            self.assertTrue(all([x in stderr for x in expected]))
        else:
            self.assertSetEqual(actual, expected)

    def testSetOverwritesOldLabelConfig(self):
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(['label', 'set', self.json_fpath, suri(bucket_uri)])
        new_key_1 = 'new_key_1'
        new_key_2 = 'new_key_2'
        new_value_1 = 'new_value_1'
        new_value_2 = 'new_value_2'
        new_json = {new_key_1: new_value_1, new_key_2: new_value_2, KEY1: 'different_value_for_an_existing_key'}
        new_json_fpath = self.CreateTempFile(contents=json.dumps(new_json).encode('ascii'))
        self.RunGsUtil(['label', 'set', new_json_fpath, suri(bucket_uri)])
        stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
        self.assertDictEqual(json.loads(stdout), new_json)

    def testInitialAndSubsequentCh(self):
        bucket_uri = self.CreateBucket()
        ch_subargs = ['-l', '%s:%s' % (KEY1, VALUE1), '-l', '%s:%s' % (KEY2, VALUE2)]
        stderr = self.RunGsUtil(['label', 'ch'] + ch_subargs + [suri(bucket_uri)], return_stderr=True)
        expected_output = _get_label_setting_output(self._use_gcloud_storage, suri(bucket_uri))
        if self._use_gcloud_storage:
            self.assertIn(expected_output, stderr)
        else:
            self.assertEqual(stderr.strip(), expected_output)
        stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
        self.assertDictEqual(json.loads(stdout), self._label_dict)
        new_key = 'new-key'
        new_value = 'new-value'
        self.RunGsUtil(['label', 'ch', '-l', '%s:%s' % (new_key, new_value), '-d', KEY2, suri(bucket_uri)])
        stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
        actual = json.loads(stdout)
        expected = {KEY1: VALUE1, new_key: new_value}
        self.assertDictEqual(actual, expected)

    def testChAppliesChangesToAllBucketArgs(self):
        bucket_suris = [suri(self.CreateBucket()), suri(self.CreateBucket())]
        ch_subargs = ['-l', '%s:%s' % (KEY1, VALUE1), '-l', '%s:%s' % (KEY2, VALUE2)]
        stderr = self.RunGsUtil(['label', 'ch'] + ch_subargs + bucket_suris, return_stderr=True)
        actual = set(stderr.splitlines())
        expected = set([_get_label_setting_output(self._use_gcloud_storage, bucket_suri) for bucket_suri in bucket_suris])
        if self._use_gcloud_storage:
            self.assertTrue(all([x in stderr for x in expected]))
        else:
            self.assertSetEqual(actual, expected)
        for bucket_suri in bucket_suris:
            stdout = self.RunGsUtil(['label', 'get', bucket_suri], return_stdout=True)
            self.assertDictEqual(json.loads(stdout), self._label_dict)

    def testChMinusDWorksWithoutExistingLabels(self):
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(['label', 'ch', '-d', 'dummy-key', suri(bucket_uri)])
        stdout = self.RunGsUtil(['label', 'get', suri(bucket_uri)], return_stdout=True)
        if self._use_gcloud_storage:
            self.assertIn('[]', stdout)
        else:
            self.assertIn('%s/ has no label configuration.' % suri(bucket_uri), stdout)

    def testTooFewArgumentsFails(self):
        """Ensures label commands fail with too few arguments."""
        invocations_missing_args = (['label'], ['label', 'set'], ['label', 'set', 'filename'], ['label', 'get'], ['label', 'ch'], ['label', 'ch', '-l', 'key:val'])
        for arg_list in invocations_missing_args:
            stderr = self.RunGsUtil(arg_list, return_stderr=True, expected_status=1)
            self.assertIn('command requires at least', stderr)
        stderr = self.RunGsUtil(['label', 'ch', 'gs://some-nonexistent-foobar-bucket-name'], return_stderr=True, expected_status=1)
        self.assertIn('Please specify at least one label change', stderr)