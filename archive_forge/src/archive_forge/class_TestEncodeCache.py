from .. import cache_utf8
from . import TestCase
class TestEncodeCache(TestCase):

    def setUp(self):
        super().setUp()
        cache_utf8.clear_encoding_cache()
        self.addCleanup(cache_utf8.clear_encoding_cache)

    def check_encode(self, rev_id):
        rev_id_utf8 = rev_id.encode('utf-8')
        self.assertFalse(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertFalse(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)
        self.assertEqual(rev_id_utf8, cache_utf8.encode(rev_id))
        self.assertTrue(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertTrue(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)
        self.assertEqual(rev_id, cache_utf8.decode(rev_id_utf8))
        cache_utf8.clear_encoding_cache()
        self.assertFalse(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertFalse(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)

    def check_decode(self, rev_id):
        rev_id_utf8 = rev_id.encode('utf-8')
        self.assertFalse(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertFalse(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)
        self.assertEqual(rev_id, cache_utf8.decode(rev_id_utf8))
        self.assertTrue(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertTrue(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)
        self.assertEqual(rev_id_utf8, cache_utf8.encode(rev_id))
        cache_utf8.clear_encoding_cache()
        self.assertFalse(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertFalse(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)

    def test_ascii(self):
        self.check_decode('all_ascii_characters123123123')
        self.check_encode('all_ascii_characters123123123')

    def test_unicode(self):
        self.check_encode('some_µ_unicode_å_chars')
        self.check_decode('some_µ_unicode_å_chars')

    def test_cached_unicode(self):
        z = 'åzz'
        x = 'µyy' + z
        y = 'µyy' + z
        self.assertIsNot(x, y)
        xp = cache_utf8.get_cached_unicode(x)
        yp = cache_utf8.get_cached_unicode(y)
        self.assertIs(xp, x)
        self.assertIs(xp, yp)

    def test_cached_utf8(self):
        x = 'µyyåzz'.encode()
        y = 'µyyåzz'.encode()
        self.assertFalse(x is y)
        xp = cache_utf8.get_cached_utf8(x)
        yp = cache_utf8.get_cached_utf8(y)
        self.assertIs(xp, x)
        self.assertIs(xp, yp)

    def test_cached_ascii(self):
        x = b'%s %s' % (b'simple', b'text')
        y = b'%s %s' % (b'simple', b'text')
        self.assertIsNot(x, y)
        xp = cache_utf8.get_cached_ascii(x)
        yp = cache_utf8.get_cached_ascii(y)
        self.assertIs(xp, x)
        self.assertIs(xp, yp)
        uni_x = cache_utf8.decode(x)
        self.assertEqual('simple text', uni_x)
        self.assertIsInstance(uni_x, str)
        utf8_x = cache_utf8.encode(uni_x)
        self.assertIs(utf8_x, x)

    def test_decode_with_None(self):
        self.assertEqual(None, cache_utf8._utf8_decode_with_None(None))
        self.assertEqual('foo', cache_utf8._utf8_decode_with_None(b'foo'))
        self.assertEqual('fµ', cache_utf8._utf8_decode_with_None(b'f\xc2\xb5'))