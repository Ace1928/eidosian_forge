import os
import stat
from dulwich.objects import Blob, Commit, Tree
from ...revision import Revision
from ...tests import TestCase, TestCaseInTempDir, UnavailableFeature
from ...transport import get_transport
from ..cache import (DictBzrGitCache, IndexBzrGitCache, IndexGitCacheFormat,
class IndexGitShaMapTests(TestCaseInTempDir, TestGitShaMap):

    def setUp(self):
        TestCaseInTempDir.setUp(self)
        transport = get_transport(self.test_dir)
        IndexGitCacheFormat().initialize(transport)
        self.cache = IndexBzrGitCache(transport)
        self.map = self.cache.idmap