import os
import stat
from dulwich.objects import Blob, Commit, Tree
from ...revision import Revision
from ...tests import TestCase, TestCaseInTempDir, UnavailableFeature
from ...transport import get_transport
from ..cache import (DictBzrGitCache, IndexBzrGitCache, IndexGitCacheFormat,
class SqliteGitShaMapTests(TestCaseInTempDir, TestGitShaMap):

    def setUp(self):
        TestCaseInTempDir.setUp(self)
        self.cache = SqliteBzrGitCache(os.path.join(self.test_dir, 'foo.db'))
        self.map = self.cache.idmap