import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
def test_set_unicode_keys(self):
    cache = self.make_file_cache()
    cache.set(self.unicode_text, b'value')
    self.assertEqual(b'value', cache.get(self.unicode_text))