from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
class MiscTest(TestCase):
    """tests various parts of utils module"""

    def test_compat(self):
        """test compat's lazymodule"""
        from passlib.utils import compat
        self.assertRegex(repr(compat), "^<module 'passlib.utils.compat' from '.*?'>$")
        dir(compat)
        self.assertTrue('UnicodeIO' in dir(compat))
        self.assertTrue('irange' in dir(compat))

    def test_classproperty(self):
        from passlib.utils.decor import classproperty

        class test(object):
            xvar = 1

            @classproperty
            def xprop(cls):
                return cls.xvar
        self.assertEqual(test.xprop, 1)
        prop = test.__dict__['xprop']
        self.assertIs(prop.im_func, prop.__func__)

    def test_deprecated_function(self):
        from passlib.utils.decor import deprecated_function

        @deprecated_function(deprecated='1.6', removed='1.8')
        def test_func(*args):
            """test docstring"""
            return args
        self.assertTrue('.. deprecated::' in test_func.__doc__)
        with self.assertWarningList(dict(category=DeprecationWarning, message='the function passlib.tests.test_utils.test_func() is deprecated as of Passlib 1.6, and will be removed in Passlib 1.8.')):
            self.assertEqual(test_func(1, 2), (1, 2))

    def test_memoized_property(self):
        from passlib.utils.decor import memoized_property

        class dummy(object):
            counter = 0

            @memoized_property
            def value(self):
                value = self.counter
                self.counter = value + 1
                return value
        d = dummy()
        self.assertEqual(d.value, 0)
        self.assertEqual(d.value, 0)
        self.assertEqual(d.counter, 1)
        prop = dummy.value
        if not PY3:
            self.assertIs(prop.im_func, prop.__func__)

    def test_getrandbytes(self):
        """getrandbytes()"""
        from passlib.utils import getrandbytes
        wrapper = partial(getrandbytes, self.getRandom())
        self.assertEqual(len(wrapper(0)), 0)
        a = wrapper(10)
        b = wrapper(10)
        self.assertIsInstance(a, bytes)
        self.assertEqual(len(a), 10)
        self.assertEqual(len(b), 10)
        self.assertNotEqual(a, b)

    @run_with_fixed_seeds(count=1024)
    def test_getrandstr(self, seed):
        """getrandstr()"""
        from passlib.utils import getrandstr
        wrapper = partial(getrandstr, self.getRandom(seed=seed))
        self.assertEqual(wrapper('abc', 0), '')
        self.assertRaises(ValueError, wrapper, 'abc', -1)
        self.assertRaises(ValueError, wrapper, '', 0)
        self.assertEqual(wrapper('a', 5), 'aaaaa')
        x = wrapper(u('abc'), 32)
        y = wrapper(u('abc'), 32)
        self.assertIsInstance(x, unicode)
        self.assertNotEqual(x, y)
        self.assertEqual(sorted(set(x)), [u('a'), u('b'), u('c')])
        x = wrapper(b'abc', 32)
        y = wrapper(b'abc', 32)
        self.assertIsInstance(x, bytes)
        self.assertNotEqual(x, y)
        self.assertEqual(sorted(set(x.decode('ascii'))), [u('a'), u('b'), u('c')])

    def test_generate_password(self):
        """generate_password()"""
        from passlib.utils import generate_password
        warnings.filterwarnings('ignore', 'The function.*generate_password\\(\\) is deprecated')
        self.assertEqual(len(generate_password(15)), 15)

    def test_is_crypt_context(self):
        """test is_crypt_context()"""
        from passlib.utils import is_crypt_context
        from passlib.context import CryptContext
        cc = CryptContext(['des_crypt'])
        self.assertTrue(is_crypt_context(cc))
        self.assertFalse(not is_crypt_context(cc))

    def test_genseed(self):
        """test genseed()"""
        import random
        from passlib.utils import genseed
        rng = random.Random(genseed())
        a = rng.randint(0, 10 ** 10)
        rng = random.Random(genseed())
        b = rng.randint(0, 10 ** 10)
        self.assertNotEqual(a, b)
        rng.seed(genseed(rng))

    def test_crypt(self):
        """test crypt.crypt() wrappers"""
        from passlib.utils import has_crypt, safe_crypt, test_crypt
        from passlib.registry import get_supported_os_crypt_schemes, get_crypt_handler
        supported = get_supported_os_crypt_schemes()
        if not has_crypt:
            self.assertEqual(supported, ())
            self.assertEqual(safe_crypt('test', 'aa'), None)
            self.assertFalse(test_crypt('test', 'aaqPiZY5xR5l.'))
            raise self.skipTest('crypt.crypt() not available')
        if not supported:
            raise self.fail('crypt() present, but no supported schemes found!')
        for scheme in ('md5_crypt', 'sha256_crypt'):
            if scheme in supported:
                break
        else:
            scheme = supported[-1]
        hasher = get_crypt_handler(scheme)
        if getattr(hasher, 'min_rounds', None):
            hasher = hasher.using(rounds=hasher.min_rounds)

        def get_hash(secret):
            assert isinstance(secret, unicode)
            hash = hasher.hash(secret)
            if isinstance(hash, bytes):
                hash = hash.decode('utf-8')
            assert isinstance(hash, unicode)
            return hash
        s1 = u('test')
        h1 = get_hash(s1)
        result = safe_crypt(s1, h1)
        self.assertIsInstance(result, unicode)
        self.assertEqual(result, h1)
        self.assertEqual(safe_crypt(to_bytes(s1), to_bytes(h1)), h1)
        h1x = h1[:-2] + 'xx'
        self.assertEqual(safe_crypt(s1, h1x), h1)
        s2 = u('testሴ')
        h2 = get_hash(s2)
        self.assertEqual(safe_crypt(s2, h2), h2)
        self.assertEqual(safe_crypt(to_bytes(s2), to_bytes(h2)), h2)
        self.assertRaises(ValueError, safe_crypt, '\x00', h1)
        self.assertTrue(test_crypt('test', h1))
        self.assertFalse(test_crypt('test', h1x))
        import passlib.utils as mod
        orig = mod._crypt
        try:
            retval = None
            mod._crypt = lambda secret, hash: retval
            for retval in [None, '', ':', ':0', '*0']:
                self.assertEqual(safe_crypt('test', h1), None)
                self.assertFalse(test_crypt('test', h1))
            retval = 'xxx'
            self.assertEqual(safe_crypt('test', h1), 'xxx')
            self.assertFalse(test_crypt('test', h1))
        finally:
            mod._crypt = orig

    def test_consteq(self):
        """test consteq()"""
        from passlib.utils import consteq, str_consteq
        self.assertRaises(TypeError, consteq, u(''), b'')
        self.assertRaises(TypeError, consteq, u(''), 1)
        self.assertRaises(TypeError, consteq, u(''), None)
        self.assertRaises(TypeError, consteq, b'', u(''))
        self.assertRaises(TypeError, consteq, b'', 1)
        self.assertRaises(TypeError, consteq, b'', None)
        self.assertRaises(TypeError, consteq, None, u(''))
        self.assertRaises(TypeError, consteq, None, b'')
        self.assertRaises(TypeError, consteq, 1, u(''))
        self.assertRaises(TypeError, consteq, 1, b'')

        def consteq_supports_string(value):
            return consteq is str_consteq or PY2 or is_ascii_safe(value)
        for value in [u('a'), u('abc'), u('ÿ¢\x12\x00') * 10]:
            if consteq_supports_string(value):
                self.assertTrue(consteq(value, value), 'value %r:' % (value,))
            else:
                self.assertRaises(TypeError, consteq, value, value)
            self.assertTrue(str_consteq(value, value), 'value %r:' % (value,))
            value = value.encode('latin-1')
            self.assertTrue(consteq(value, value), 'value %r:' % (value,))
        for l, r in [(u('a'), u('c')), (u('abcabc'), u('zbaabc')), (u('abcabc'), u('abzabc')), (u('abcabc'), u('abcabz')), ((u('ÿ¢\x12\x00') * 10)[:-1] + u('\x01'), u('ÿ¢\x12\x00') * 10), (u(''), u('a')), (u('abc'), u('abcdef')), (u('abc'), u('defabc')), (u('qwertyuiopasdfghjklzxcvbnm'), u('abc'))]:
            if consteq_supports_string(l) and consteq_supports_string(r):
                self.assertFalse(consteq(l, r), 'values %r %r:' % (l, r))
                self.assertFalse(consteq(r, l), 'values %r %r:' % (r, l))
            else:
                self.assertRaises(TypeError, consteq, l, r)
                self.assertRaises(TypeError, consteq, r, l)
            self.assertFalse(str_consteq(l, r), 'values %r %r:' % (l, r))
            self.assertFalse(str_consteq(r, l), 'values %r %r:' % (r, l))
            l = l.encode('latin-1')
            r = r.encode('latin-1')
            self.assertFalse(consteq(l, r), 'values %r %r:' % (l, r))
            self.assertFalse(consteq(r, l), 'values %r %r:' % (r, l))

    def test_saslprep(self):
        """test saslprep() unicode normalizer"""
        self.require_stringprep()
        from passlib.utils import saslprep as sp
        self.assertRaises(TypeError, sp, None)
        self.assertRaises(TypeError, sp, 1)
        self.assertRaises(TypeError, sp, b'')
        self.assertEqual(sp(u('')), u(''))
        self.assertEqual(sp(u('\xad')), u(''))
        self.assertEqual(sp(u('$\xad$\u200d$')), u('$$$'))
        self.assertEqual(sp(u('$ $\xa0$\u3000$')), u('$ $ $ $'))
        self.assertEqual(sp(u('à')), u('à'))
        self.assertEqual(sp(u('à')), u('à'))
        self.assertRaises(ValueError, sp, u('\x00'))
        self.assertRaises(ValueError, sp, u('\x7f'))
        self.assertRaises(ValueError, sp, u('\u180e'))
        self.assertRaises(ValueError, sp, u('\ufff9'))
        self.assertRaises(ValueError, sp, u('\ue000'))
        self.assertRaises(ValueError, sp, u('\ufdd0'))
        self.assertRaises(ValueError, sp, u('\ud800'))
        self.assertRaises(ValueError, sp, u('�'))
        self.assertRaises(ValueError, sp, u('⿰'))
        self.assertRaises(ValueError, sp, u('\u200e'))
        self.assertRaises(ValueError, sp, u('\u206f'))
        self.assertRaises(ValueError, sp, u('ऀ'))
        self.assertRaises(ValueError, sp, u('\ufff8'))
        self.assertRaises(ValueError, sp, u('\U000e0001'))
        self.assertRaises(ValueError, sp, u('ا1'))
        self.assertEqual(sp(u('ا')), u('ا'))
        self.assertEqual(sp(u('اب')), u('اب'))
        self.assertEqual(sp(u('ا1ب')), u('ا1ب'))
        self.assertRaises(ValueError, sp, u('اAب'))
        self.assertRaises(ValueError, sp, u('xاz'))
        self.assertEqual(sp(u('xAz')), u('xAz'))
        self.assertEqual(sp(u('I\xadX')), u('IX'))
        self.assertEqual(sp(u('user')), u('user'))
        self.assertEqual(sp(u('USER')), u('USER'))
        self.assertEqual(sp(u('ª')), u('a'))
        self.assertEqual(sp(u('Ⅸ')), u('IX'))
        self.assertRaises(ValueError, sp, u('\x07'))
        self.assertRaises(ValueError, sp, u('ا1'))
        self.assertRaises(ValueError, sp, u('ا1'))
        self.assertEqual(sp(u('ا1ب')), u('ا1ب'))

    def test_splitcomma(self):
        from passlib.utils import splitcomma
        self.assertEqual(splitcomma(''), [])
        self.assertEqual(splitcomma(','), [])
        self.assertEqual(splitcomma('a'), ['a'])
        self.assertEqual(splitcomma(' a , '), ['a'])
        self.assertEqual(splitcomma(' a , b'), ['a', 'b'])
        self.assertEqual(splitcomma(' a, b, '), ['a', 'b'])

    def test_utf8_truncate(self):
        """
        utf8_truncate()
        """
        from passlib.utils import utf8_truncate
        for source in [b'', b'1', b'123', b'\x1a', b'\x1a' * 10, b'\x7f', b'\x7f' * 10, b'a\xc2\xa0\xc3\xbe\xc3\xbe', b'abcdefghjusdfaoiu\xc2\xa0\xc3\xbe\xc3\xbedsfioauweoiruer']:
            source.decode('utf-8')
            end = len(source)
            for idx in range(end + 16):
                prefix = 'source=%r index=%r: ' % (source, idx)
                result = utf8_truncate(source, idx)
                result.decode('utf-8')
                self.assertLessEqual(len(result), end, msg=prefix)
                self.assertGreaterEqual(len(result), min(idx, end), msg=prefix)
                self.assertLess(len(result), min(idx + 4, end + 1), msg=prefix)
                self.assertEqual(result, source[:len(result)], msg=prefix)
        for source in [b'\xca', b'\xca' * 10, b'\x00', b'\x00' * 10]:
            end = len(source)
            for idx in range(end + 16):
                prefix = 'source=%r index=%r: ' % (source, idx)
                result = utf8_truncate(source, idx)
                self.assertEqual(result, source[:idx], msg=prefix)
        for source in [b'\xaa', b'\xaa' * 10]:
            end = len(source)
            for idx in range(end + 16):
                prefix = 'source=%r index=%r: ' % (source, idx)
                result = utf8_truncate(source, idx)
                self.assertEqual(result, source[:idx + 3], msg=prefix)
        source = b'MN\xff\xa0\xa1\xa2\xaaOP\xab'
        self.assertEqual(utf8_truncate(source, 0), b'')
        self.assertEqual(utf8_truncate(source, 1), b'M')
        self.assertEqual(utf8_truncate(source, 2), b'MN')
        self.assertEqual(utf8_truncate(source, 3), b'MN\xff\xa0\xa1\xa2')
        self.assertEqual(utf8_truncate(source, 4), b'MN\xff\xa0\xa1\xa2\xaa')
        self.assertEqual(utf8_truncate(source, 5), b'MN\xff\xa0\xa1\xa2\xaa')
        self.assertEqual(utf8_truncate(source, 6), b'MN\xff\xa0\xa1\xa2\xaa')
        self.assertEqual(utf8_truncate(source, 7), b'MN\xff\xa0\xa1\xa2\xaa')
        self.assertEqual(utf8_truncate(source, 8), b'MN\xff\xa0\xa1\xa2\xaaO')
        self.assertEqual(utf8_truncate(source, 9), b'MN\xff\xa0\xa1\xa2\xaaOP\xab')
        self.assertEqual(utf8_truncate(source, 10), b'MN\xff\xa0\xa1\xa2\xaaOP\xab')
        self.assertEqual(utf8_truncate(source, 11), b'MN\xff\xa0\xa1\xa2\xaaOP\xab')