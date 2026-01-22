import os
import sys
from .. import bedding, osutils, tests
class TestXDGCacheDir(tests.TestCaseInTempDir):

    def setUp(self):
        super().setUp()
        if sys.platform in ('darwin', 'win32'):
            raise tests.TestNotApplicable('XDG cache dir not used on this platform')
        self.overrideEnv('HOME', self.test_home_dir)
        self.overrideEnv('BZR_HOME', None)

    def test_xdg_cache_dir_exists(self):
        """When ~/.cache/breezy exists, use it as the cache dir."""
        cachedir = osutils.pathjoin(self.test_home_dir, '.cache')
        newdir = osutils.pathjoin(cachedir, 'breezy')
        self.assertEqual(bedding.cache_dir(), newdir)

    def test_xdg_cache_home_unix(self):
        """When XDG_CACHE_HOME is set, use it."""
        if sys.platform in ('nt', 'win32'):
            raise tests.TestNotApplicable('XDG cache dir not used on this platform')
        xdgcachedir = osutils.pathjoin(self.test_home_dir, 'xdgcache')
        self.overrideEnv('XDG_CACHE_HOME', xdgcachedir)
        newdir = osutils.pathjoin(xdgcachedir, 'breezy')
        self.assertEqual(bedding.cache_dir(), newdir)