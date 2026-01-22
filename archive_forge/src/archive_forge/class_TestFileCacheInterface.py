import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
class TestFileCacheInterface(unittest.TestCase):
    """Tests for ``AtomicFileCache``."""
    file_cache_factory = httplib2.FileCache
    unicode_bytes = b'pa\xc9\xaa\xce\xb8\xc9\x99n'
    unicode_text = unicode_bytes.decode('utf-8')

    def setUp(self):
        super(TestFileCacheInterface, self).setUp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.cache_dir)
        super(TestFileCacheInterface, self).tearDown()

    def make_file_cache(self):
        """Make a FileCache-like object to be tested."""
        return self.file_cache_factory(self.cache_dir, safename)

    def test_get_non_existent_key(self):
        cache = self.make_file_cache()
        self.assertIs(None, cache.get('nonexistent'))

    def test_set_key(self):
        cache = self.make_file_cache()
        cache.set('key', b'value')
        self.assertEqual(b'value', cache.get('key'))

    def test_set_twice_overrides(self):
        cache = self.make_file_cache()
        cache.set('key', b'value')
        cache.set('key', b'new-value')
        self.assertEqual(b'new-value', cache.get('key'))

    def test_delete_absent_key(self):
        cache = self.make_file_cache()
        cache.delete('nonexistent')
        self.assertIs(None, cache.get('nonexistent'))

    def test_delete_key(self):
        cache = self.make_file_cache()
        cache.set('key', b'value')
        cache.delete('key')
        self.assertIs(None, cache.get('key'))

    def test_get_non_string_key(self):
        cache = self.make_file_cache()
        self.assertRaises(TypeError, cache.get, 42)

    def test_delete_non_string_key(self):
        cache = self.make_file_cache()
        self.assertRaises(TypeError, cache.delete, 42)

    def test_set_non_string_key(self):
        cache = self.make_file_cache()
        self.assertRaises(TypeError, cache.set, 42, 'the answer')

    def test_set_non_string_value(self):
        cache = self.make_file_cache()
        self.assertRaises(TypeError, cache.set, 'answer', 42)
        self.assertEqual(b'', cache.get('answer'))

    def test_get_unicode(self):
        cache = self.make_file_cache()
        self.assertIs(None, cache.get(self.unicode_text))

    def test_set_unicode_keys(self):
        cache = self.make_file_cache()
        cache.set(self.unicode_text, b'value')
        self.assertEqual(b'value', cache.get(self.unicode_text))

    def test_set_unicode_value(self):
        cache = self.make_file_cache()
        error = TypeError if PY3 else UnicodeEncodeError
        self.assertRaises(error, cache.set, 'key', self.unicode_text)

    def test_delete_unicode(self):
        cache = self.make_file_cache()
        cache.set(self.unicode_text, b'value')
        cache.delete(self.unicode_text)
        self.assertIs(None, cache.get(self.unicode_text))