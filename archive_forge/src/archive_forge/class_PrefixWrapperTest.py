from __future__ import with_statement
import re
import hashlib
from logging import getLogger
import warnings
from passlib.hash import ldap_md5, sha256_crypt
from passlib.exc import MissingBackendError, PasslibHashWarning
from passlib.utils.compat import str_to_uascii, \
import passlib.utils.handlers as uh
from passlib.tests.utils import HandlerCase, TestCase
from passlib.utils.compat import u
class PrefixWrapperTest(TestCase):
    """test PrefixWrapper class"""

    def test_00_lazy_loading(self):
        """test PrefixWrapper lazy loading of handler"""
        d1 = uh.PrefixWrapper('d1', 'ldap_md5', '{XXX}', '{MD5}', lazy=True)
        self.assertEqual(d1._wrapped_name, 'ldap_md5')
        self.assertIs(d1._wrapped_handler, None)
        self.assertIs(d1.wrapped, ldap_md5)
        self.assertIs(d1._wrapped_handler, ldap_md5)
        with dummy_handler_in_registry('ldap_md5') as dummy:
            self.assertIs(d1.wrapped, ldap_md5)

    def test_01_active_loading(self):
        """test PrefixWrapper active loading of handler"""
        d1 = uh.PrefixWrapper('d1', 'ldap_md5', '{XXX}', '{MD5}')
        self.assertEqual(d1._wrapped_name, 'ldap_md5')
        self.assertIs(d1._wrapped_handler, ldap_md5)
        self.assertIs(d1.wrapped, ldap_md5)
        with dummy_handler_in_registry('ldap_md5') as dummy:
            self.assertIs(d1.wrapped, ldap_md5)

    def test_02_explicit(self):
        """test PrefixWrapper with explicitly specified handler"""
        d1 = uh.PrefixWrapper('d1', ldap_md5, '{XXX}', '{MD5}')
        self.assertEqual(d1._wrapped_name, None)
        self.assertIs(d1._wrapped_handler, ldap_md5)
        self.assertIs(d1.wrapped, ldap_md5)
        with dummy_handler_in_registry('ldap_md5') as dummy:
            self.assertIs(d1.wrapped, ldap_md5)

    def test_10_wrapped_attributes(self):
        d1 = uh.PrefixWrapper('d1', 'ldap_md5', '{XXX}', '{MD5}')
        self.assertEqual(d1.name, 'd1')
        self.assertIs(d1.setting_kwds, ldap_md5.setting_kwds)
        self.assertFalse('max_rounds' in dir(d1))
        d2 = uh.PrefixWrapper('d2', 'sha256_crypt', '{XXX}')
        self.assertIs(d2.setting_kwds, sha256_crypt.setting_kwds)
        self.assertTrue('max_rounds' in dir(d2))

    def test_11_wrapped_methods(self):
        d1 = uh.PrefixWrapper('d1', 'ldap_md5', '{XXX}', '{MD5}')
        dph = '{XXX}X03MO1qnZdYdgyfeuILPmQ=='
        lph = '{MD5}X03MO1qnZdYdgyfeuILPmQ=='
        self.assertEqual(d1.genconfig(), '{XXX}1B2M2Y8AsgTpgAmY7PhCfg==')
        self.assertRaises(TypeError, d1.genhash, 'password', None)
        self.assertEqual(d1.genhash('password', dph), dph)
        self.assertRaises(ValueError, d1.genhash, 'password', lph)
        self.assertEqual(d1.hash('password'), dph)
        self.assertTrue(d1.identify(dph))
        self.assertFalse(d1.identify(lph))
        self.assertRaises(ValueError, d1.verify, 'password', lph)
        self.assertTrue(d1.verify('password', dph))

    def test_12_ident(self):
        h = uh.PrefixWrapper('h2', 'ldap_md5', '{XXX}')
        self.assertEqual(h.ident, u('{XXX}{MD5}'))
        self.assertIs(h.ident_values, None)
        h = uh.PrefixWrapper('h2', 'des_crypt', '{XXX}')
        self.assertIs(h.ident, None)
        self.assertIs(h.ident_values, None)
        h = uh.PrefixWrapper('h1', 'ldap_md5', '{XXX}', '{MD5}')
        self.assertIs(h.ident, None)
        self.assertIs(h.ident_values, None)
        h = uh.PrefixWrapper('h3', 'ldap_md5', '{XXX}', ident='{X')
        self.assertEqual(h.ident, u('{X'))
        self.assertIs(h.ident_values, None)
        h = uh.PrefixWrapper('h3', 'ldap_md5', '{XXX}', ident='{XXX}A')
        self.assertRaises(ValueError, uh.PrefixWrapper, 'h3', 'ldap_md5', '{XXX}', ident='{XY')
        self.assertRaises(ValueError, uh.PrefixWrapper, 'h3', 'ldap_md5', '{XXX}', ident='{XXXX')
        h = uh.PrefixWrapper('h4', 'phpass', '{XXX}')
        self.assertIs(h.ident, None)
        self.assertEqual(h.ident_values, (u('{XXX}$P$'), u('{XXX}$H$')))
        h = uh.PrefixWrapper('h5', 'des_crypt', '{XXX}', ident=True)
        self.assertEqual(h.ident, u('{XXX}'))
        self.assertIs(h.ident_values, None)
        self.assertRaises(ValueError, uh.PrefixWrapper, 'h6', 'des_crypt', ident=True)
        with self.assertWarningList('orig_prefix.*may not work correctly'):
            h = uh.PrefixWrapper('h7', 'phpass', orig_prefix='$', prefix='?')
        self.assertEqual(h.ident_values, None)
        self.assertEqual(h.ident, None)

    def test_13_repr(self):
        """test repr()"""
        h = uh.PrefixWrapper('h2', 'md5_crypt', '{XXX}', orig_prefix='$1$')
        self.assertRegex(repr(h), '(?x)^PrefixWrapper\\(\n                [\'"]h2[\'"],\\s+\n                [\'"]md5_crypt[\'"],\\s+\n                prefix=u?["\']{XXX}[\'"],\\s+\n                orig_prefix=u?["\']\\$1\\$[\'"]\n            \\)$')

    def test_14_bad_hash(self):
        """test orig_prefix sanity check"""
        h = uh.PrefixWrapper('h2', 'md5_crypt', orig_prefix='$6$')
        self.assertRaises(ValueError, h.hash, 'test')