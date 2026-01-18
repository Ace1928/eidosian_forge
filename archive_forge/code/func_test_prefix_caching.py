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
def test_prefix_caching(self):
    """Tests tab completion results returned from cache with prefix match.

    If the tab completion prefix is an extension of the cached prefix, tab
    completion should return results from the cache that start with the prefix.
    """
    with SetBotoConfigForTest([('GSUtil', 'state_dir', self.CreateTempDir())]):
        cached_prefix = 'gs://prefix'
        cached_results = ['gs://prefix-first', 'gs://prefix-second']
        _WriteTabCompletionCache(cached_prefix, cached_results)
        request = 'gs://prefix-f'
        completer = CloudObjectCompleter(self.MakeGsUtilApi())
        results = completer(request)
        self.assertEqual(['gs://prefix-first'], results)