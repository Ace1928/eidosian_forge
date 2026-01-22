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