from __future__ import absolute_import, division, print_function
import logging; log = logging.getLogger(__name__)
import sys
import re
from passlib import apps as _apps, exc, registry
from passlib.apps import django10_context, django14_context, django16_context
from passlib.context import CryptContext
from passlib.ext.django.utils import (
from passlib.utils.compat import iteritems, get_method_function, u
from passlib.utils.decor import memoized_property
from passlib.tests.utils import TestCase, TEST_MODE, handler_derived_from
from passlib.tests.test_handlers import get_handler_case
from passlib.hash import django_pbkdf2_sha256
class DjangoBehaviorTest(_ExtensionTest):
    """
    tests model to verify it matches django's behavior.

    running this class verifies the tests correctly assert what Django itself does.

    running the ExtensionBehaviorTest subclass below verifies "passlib.ext.django"
    matches what the tests assert.
    """
    descriptionPrefix = 'verify django behavior'
    patched = False
    config = stock_config

    @memoized_property
    def context(self):
        """
        per-test CryptContext() created from .config.
        """
        return CryptContext._norm_source(self.config)

    def assert_unusable_password(self, user):
        """
        check that user object is set to 'unusable password' constant
        """
        self.assertTrue(user.password.startswith('!'))
        self.assertFalse(user.has_usable_password())
        self.assertEqual(user.pop_saved_passwords(), [])

    def assert_valid_password(self, user, hash=UNSET, saved=None):
        """
        check that user object has a usable password hash.
        :param hash: optionally check it has this exact hash
        :param saved: check that mock commit history for user.password matches this list
        """
        if hash is UNSET:
            self.assertNotEqual(user.password, '!')
            self.assertNotEqual(user.password, None)
        else:
            self.assertEqual(user.password, hash)
        self.assertTrue(user.has_usable_password(), 'hash should be usable: %r' % (user.password,))
        self.assertEqual(user.pop_saved_passwords(), [] if saved is None else [saved])

    def test_extension_config(self):
        """
        test extension config is loaded correctly
        """
        if not self.patched:
            raise self.skipTest('extension not loaded')
        ctx = self.context
        from django.contrib.auth.hashers import check_password
        from passlib.ext.django.models import password_context
        self.assertEqual(password_context.to_dict(resolve=True), ctx.to_dict(resolve=True))
        from django.contrib.auth.models import check_password as check_password2
        self.assertEqual(check_password2, check_password)

    def test_default_algorithm(self):
        """
        test django's default algorithm
        """
        ctx = self.context
        from django.contrib.auth.hashers import make_password
        user = FakeUser()
        user.set_password(PASS1)
        self.assertTrue(ctx.handler().verify(PASS1, user.password))
        self.assert_valid_password(user)
        hash = make_password(PASS1)
        self.assertTrue(ctx.handler().verify(PASS1, hash))

    def test_empty_password(self):
        """
        test how methods handle empty string as password
        """
        ctx = self.context
        from django.contrib.auth.hashers import check_password, make_password, is_password_usable, identify_hasher
        user = FakeUser()
        user.set_password('')
        hash = user.password
        self.assertTrue(ctx.handler().verify('', hash))
        self.assert_valid_password(user, hash)
        self.assertTrue(user.check_password(''))
        self.assert_valid_password(user, hash)
        self.assertTrue(check_password('', hash))

    def test_unusable_flag(self):
        """
        test how methods handle 'unusable flag' in hash
        """
        from django.contrib.auth.hashers import check_password, make_password, is_password_usable, identify_hasher
        user = FakeUser()
        user.set_unusable_password()
        self.assert_unusable_password(user)
        user = FakeUser()
        user.set_password(None)
        self.assert_unusable_password(user)
        self.assertFalse(user.check_password(None))
        self.assertFalse(user.check_password('None'))
        self.assertFalse(user.check_password(''))
        self.assertFalse(user.check_password(PASS1))
        self.assertFalse(user.check_password(WRONG1))
        self.assert_unusable_password(user)
        self.assertTrue(make_password(None).startswith('!'))
        self.assertFalse(check_password(PASS1, '!'))
        self.assertFalse(is_password_usable(user.password))
        self.assertRaises(ValueError, identify_hasher, user.password)

    def test_none_hash_value(self):
        """
        test how methods handle None as hash value
        """
        patched = self.patched
        from django.contrib.auth.hashers import check_password, make_password, is_password_usable, identify_hasher
        user = FakeUser()
        user.password = None
        if quirks.none_causes_check_password_error and (not patched):
            self.assertRaises(TypeError, user.check_password, PASS1)
        else:
            self.assertFalse(user.check_password(PASS1))
        self.assertEqual(user.has_usable_password(), quirks.empty_is_usable_password)
        if quirks.none_causes_check_password_error and (not patched):
            self.assertRaises(TypeError, check_password, PASS1, None)
        else:
            self.assertFalse(check_password(PASS1, None))
        self.assertRaises(TypeError, identify_hasher, None)

    def test_empty_hash_value(self):
        """
        test how methods handle empty string as hash value
        """
        from django.contrib.auth.hashers import check_password, make_password, is_password_usable, identify_hasher
        user = FakeUser()
        user.password = ''
        self.assertFalse(user.check_password(PASS1))
        self.assertEqual(user.password, '')
        self.assertEqual(user.pop_saved_passwords(), [])
        self.assertEqual(user.has_usable_password(), quirks.empty_is_usable_password)
        self.assertFalse(check_password(PASS1, ''))
        self.assertRaises(ValueError, identify_hasher, '')

    def test_invalid_hash_values(self):
        """
        test how methods handle invalid hash values.
        """
        for hash in ['$789$foo']:
            with self.subTest(hash=hash):
                self._do_test_invalid_hash_value(hash)

    def _do_test_invalid_hash_value(self, hash):
        from django.contrib.auth.hashers import check_password, make_password, is_password_usable, identify_hasher
        user = FakeUser()
        user.password = hash
        self.assertFalse(user.check_password(PASS1))
        self.assertEqual(user.password, hash)
        self.assertEqual(user.pop_saved_passwords(), [])
        self.assertEqual(user.has_usable_password(), quirks.invalid_is_usable_password)
        self.assertFalse(check_password(PASS1, hash))
        self.assertRaises(ValueError, identify_hasher, hash)

    def test_available_schemes(self):
        """
        run a bunch of subtests for each hasher available in the default django setup
        (as determined by reading self.context)
        """
        for scheme in self.context.schemes():
            with self.subTest(scheme=scheme):
                self._do_test_available_scheme(scheme)

    def _do_test_available_scheme(self, scheme):
        """
        helper to test how specific hasher behaves.
        :param scheme: *passlib* name of hasher (e.g. "django_pbkdf2_sha256")
        """
        log = self.getLogger()
        ctx = self.context
        patched = self.patched
        setter = create_mock_setter()
        from django.contrib.auth.hashers import check_password, make_password, is_password_usable, identify_hasher
        handler = ctx.handler(scheme)
        log.debug('testing scheme: %r => %r', scheme, handler)
        deprecated = ctx.handler(scheme).deprecated
        assert not deprecated or scheme != ctx.default_scheme()
        try:
            testcase = get_handler_case(scheme)
        except exc.MissingBackendError:
            raise self.skipTest('backend not available')
        assert handler_derived_from(handler, testcase.handler)
        if handler.is_disabled:
            raise self.skipTest('skip disabled hasher')
        if not patched and (not check_django_hasher_has_backend(handler.django_name)):
            assert scheme in ['django_bcrypt', 'django_bcrypt_sha256', 'django_argon2'], '%r scheme should always have active backend' % scheme
            log.warning('skipping scheme %r due to missing django dependency', scheme)
            raise self.skipTest('skip due to missing dependency')
        try:
            secret, hash = sample_hashes[scheme]
        except KeyError:
            get_sample_hash = testcase('setUp').get_sample_hash
            while True:
                secret, hash = get_sample_hash()
                if secret:
                    break
        other = 'dontletmein'
        user = FakeUser()
        user.password = hash
        self.assertFalse(user.check_password(None))
        self.assertFalse(user.check_password(other))
        self.assert_valid_password(user, hash)
        self.assertTrue(user.check_password(secret))
        needs_update = deprecated
        if needs_update:
            self.assertNotEqual(user.password, hash)
            self.assertFalse(handler.identify(user.password))
            self.assertTrue(ctx.handler().verify(secret, user.password))
            self.assert_valid_password(user, saved=user.password)
        else:
            self.assert_valid_password(user, hash)
        if TEST_MODE(max='default'):
            return
        alg = DjangoTranslator().passlib_to_django_name(scheme)
        hash2 = make_password(secret, hasher=alg)
        self.assertTrue(handler.verify(secret, hash2))
        self.assertTrue(check_password(secret, hash, setter=setter))
        self.assertEqual(setter.popstate(), [secret] if needs_update else [])
        self.assertFalse(check_password(other, hash, setter=setter))
        self.assertEqual(setter.popstate(), [])
        self.assertTrue(is_password_usable(hash))
        name = DjangoTranslator().django_to_passlib_name(identify_hasher(hash).algorithm)
        self.assertEqual(name, scheme)