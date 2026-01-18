import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
def test_set_unicode_value(self):
    cache = self.make_file_cache()
    error = TypeError if PY3 else UnicodeEncodeError
    self.assertRaises(error, cache.set, 'key', self.unicode_text)