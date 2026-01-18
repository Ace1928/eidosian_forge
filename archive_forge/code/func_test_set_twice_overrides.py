import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
def test_set_twice_overrides(self):
    cache = self.make_file_cache()
    cache.set('key', b'value')
    cache.set('key', b'new-value')
    self.assertEqual(b'new-value', cache.get('key'))