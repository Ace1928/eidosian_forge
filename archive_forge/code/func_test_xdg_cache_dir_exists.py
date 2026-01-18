import os
import sys
from .. import bedding, osutils, tests
def test_xdg_cache_dir_exists(self):
    """When ~/.cache/breezy exists, use it as the cache dir."""
    cachedir = osutils.pathjoin(self.test_home_dir, '.cache')
    newdir = osutils.pathjoin(cachedir, 'breezy')
    self.assertEqual(bedding.cache_dir(), newdir)