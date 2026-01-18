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