from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import time
from gslib.command import CreateOrGetGsutilLogger
from gslib.tab_complete import CloudObjectCompleter
from gslib.tab_complete import TAB_COMPLETE_CACHE_TTL
from gslib.tab_complete import TabCompletionCache
import gslib.tests.testcase as testcase
from gslib.tests.util import ARGCOMPLETE_AVAILABLE
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.tests.util import WorkingDirectory
from gslib.utils.boto_util import GetTabCompletionCacheFilename
@unittest.skipUnless(ARGCOMPLETE_AVAILABLE, 'Tab completion requires argcomplete')
class TestTabComplete(testcase.GsUtilIntegrationTestCase):
    """Integration tests for tab completion."""

    def setUp(self):
        super(TestTabComplete, self).setUp()
        self.logger = CreateOrGetGsutilLogger('tab_complete')

    def test_single_bucket(self):
        """Tests tab completion matching a single bucket."""
        bucket_name = self.MakeTempName('bucket', prefix='aaa-')
        self.CreateBucket(bucket_name)
        request = '%s://%s' % (self.default_provider, bucket_name[:-2])
        expected_result = '//%s/' % bucket_name
        self.RunGsUtilTabCompletion(['ls', request], expected_results=[expected_result])

    def test_bucket_only_single_bucket(self):
        """Tests bucket-only tab completion matching a single bucket."""
        bucket_name = self.MakeTempName('bucket', prefix='aaa-')
        self.CreateBucket(bucket_name)
        request = '%s://%s' % (self.default_provider, bucket_name[:-2])
        expected_result = '//%s ' % bucket_name
        self.RunGsUtilTabCompletion(['rb', request], expected_results=[expected_result])

    def test_bucket_only_no_objects(self):
        """Tests that bucket-only tab completion doesn't match objects."""
        object_name = self.MakeTempName('obj')
        object_uri = self.CreateObject(object_name=object_name, contents=b'data')
        request = '%s://%s/%s' % (self.default_provider, object_uri.bucket_name, object_name[:-2])
        self.RunGsUtilTabCompletion(['rb', request], expected_results=[])

    def test_single_subdirectory(self):
        """Tests tab completion matching a single subdirectory."""
        object_base_name = self.MakeTempName('obj')
        object_name = object_base_name + '/subobj'
        object_uri = self.CreateObject(object_name=object_name, contents=b'data')
        request = '%s://%s/' % (self.default_provider, object_uri.bucket_name)
        expected_result = '//%s/%s/' % (object_uri.bucket_name, object_base_name)
        self.RunGsUtilTabCompletion(['ls', request], expected_results=[expected_result])

    def test_multiple_buckets(self):
        """Tests tab completion matching multiple buckets."""
        base_name = self.MakeTempName('bucket')
        prefix = 'aaa-'
        self.CreateBucket(base_name, bucket_name_prefix=prefix, bucket_name_suffix='1')
        self.CreateBucket(base_name, bucket_name_prefix=prefix, bucket_name_suffix='2')
        request = '%s://%s' % (self.default_provider, ''.join([prefix, base_name]))
        expected_result1 = '//%s/' % ''.join([prefix, base_name, '1'])
        expected_result2 = '//%s/' % ''.join([prefix, base_name, '2'])
        self.RunGsUtilTabCompletion(['ls', request], expected_results=[expected_result1, expected_result2])

    def test_single_object(self):
        """Tests tab completion matching a single object."""
        object_name = self.MakeTempName('obj')
        object_uri = self.CreateObject(object_name=object_name, contents=b'data')
        request = '%s://%s/%s' % (self.default_provider, object_uri.bucket_name, object_name[:-2])
        expected_result = '//%s/%s ' % (object_uri.bucket_name, object_name)
        self.RunGsUtilTabCompletion(['ls', request], expected_results=[expected_result])

    def test_multiple_objects(self):
        """Tests tab completion matching multiple objects."""
        bucket_uri = self.CreateBucket()
        object_base_name = self.MakeTempName('obj')
        object1_name = object_base_name + '-suffix1'
        self.CreateObject(bucket_uri=bucket_uri, object_name=object1_name, contents=b'data')
        object2_name = object_base_name + '-suffix2'
        self.CreateObject(bucket_uri=bucket_uri, object_name=object2_name, contents=b'data')
        request = '%s://%s/%s' % (self.default_provider, bucket_uri.bucket_name, object_base_name)
        expected_result1 = '//%s/%s' % (bucket_uri.bucket_name, object1_name)
        expected_result2 = '//%s/%s' % (bucket_uri.bucket_name, object2_name)
        self.RunGsUtilTabCompletion(['ls', request], expected_results=[expected_result1, expected_result2])

    def test_subcommands(self):
        """Tests tab completion for commands with subcommands."""
        bucket_name = self.MakeTempName('bucket', prefix='aaa-')
        self.CreateBucket(bucket_name)
        bucket_request = '%s://%s' % (self.default_provider, bucket_name[:-2])
        expected_bucket_result = '//%s ' % bucket_name
        local_file = 'a_local_file'
        local_dir = self.CreateTempDir(test_files=[local_file])
        local_file_request = '%s%s' % (local_dir, os.sep)
        expected_local_file_result = '%s ' % os.path.join(local_dir, local_file)
        self.RunGsUtilTabCompletion(['cors', 'get', bucket_request], expected_results=[expected_bucket_result])
        self.RunGsUtilTabCompletion(['cors', 'set', local_file_request], expected_results=[expected_local_file_result])
        self.RunGsUtilTabCompletion(['cors', 'set', 'some_file', bucket_request], expected_results=[expected_bucket_result])

    def test_invalid_partial_bucket_name(self):
        """Tests tab completion with a partial URL that by itself is not valid.

    The bucket name in a Cloud URL cannot end in a dash, but a partial URL
    during tab completion may end in a dash and completion should still work.
    """
        bucket_base_name = self.MakeTempName('bucket', prefix='aaa-')
        bucket_name = bucket_base_name + '-s'
        self.CreateBucket(bucket_name)
        request = '%s://%s-' % (self.default_provider, bucket_base_name)
        expected_result = '//%s/' % bucket_name
        self.RunGsUtilTabCompletion(['ls', request], expected_results=[expected_result])

    def test_acl_argument(self):
        """Tests tab completion for ACL arguments."""
        local_file = 'a_local_file'
        local_dir = self.CreateTempDir(test_files=[local_file])
        local_file_request = '%s%s' % (local_dir, os.sep)
        expected_local_file_result = '%s ' % os.path.join(local_dir, local_file)
        self.RunGsUtilTabCompletion(['acl', 'set', local_file_request], expected_results=[expected_local_file_result])
        self.RunGsUtilTabCompletion(['acl', 'set', 'priv'], expected_results=['private '])
        local_file = 'priv_file'
        local_dir = self.CreateTempDir(test_files=[local_file])
        with WorkingDirectory(local_dir):
            self.RunGsUtilTabCompletion(['acl', 'set', 'priv'], expected_results=[local_file, 'private'])