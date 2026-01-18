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
def test_prefix_caching_partial_results(self):
    """Tests tab completion prefix matching ignoring partial cached results.

    If the tab completion prefix is an extension of the cached prefix, but the
    cached result set is partial, the cached results should not be used because
    the matching results for the prefix may be incomplete.
    """
    with SetBotoConfigForTest([('GSUtil', 'state_dir', self.CreateTempDir())]):
        object_uri = self.CreateObject(object_name='obj', contents=b'test data')
        cached_prefix = '%s://%s/' % (self.default_provider, object_uri.bucket_name)
        cached_results = []
        _WriteTabCompletionCache(cached_prefix, cached_results, partial_results=True)
        request = '%s://%s/o' % (self.default_provider, object_uri.bucket_name)
        completer = CloudObjectCompleter(self.MakeGsUtilApi())
        results = completer(request)
        self.assertEqual([str(object_uri)], results)