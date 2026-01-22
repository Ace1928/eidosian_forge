from __future__ import with_statement, absolute_import
from base64 import b64encode
from hashlib import sha256
import os
import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.crypto.digest import compile_hmac
from passlib.exc import PasslibHashWarning, PasslibSecurityWarning, PasslibSecurityError
from passlib.utils import safe_crypt, repeat_string, to_bytes, parse_version, \
from passlib.utils.binary import bcrypt64
from passlib.utils.compat import get_unbound_method_function
from passlib.utils.compat import u, uascii_to_str, unicode, str_to_uascii, PY3, error_from
import passlib.utils.handlers as uh
@classmethod
def _finalize_backend_mixin(mixin_cls, backend, dryrun):
    """
        helper called by from backend mixin classes' _load_backend_mixin() --
        invoked after backend imports have been loaded, and performs
        feature detection & testing common to all backends.
        """
    assert mixin_cls is bcrypt._backend_mixin_map[backend], '_configure_workarounds() invoked from wrong class'
    if mixin_cls._workrounds_initialized:
        return True
    verify = mixin_cls.verify
    err_types = (ValueError, uh.exc.MissingBackendError)
    if _bcryptor:
        err_types += (_bcryptor.engine.SaltError,)

    def safe_verify(secret, hash):
        """verify() wrapper which traps 'unknown identifier' errors"""
        try:
            return verify(secret, hash)
        except err_types:
            return NotImplemented
        except uh.exc.InternalBackendError:
            log.debug('trapped unexpected response from %r backend: verify(%r, %r):', backend, secret, hash, exc_info=True)
            return NotImplemented

    def assert_lacks_8bit_bug(ident):
        """
            helper to check for cryptblowfish 8bit bug (fixed in 2y/2b);
            even though it's not known to be present in any of passlib's backends.
            this is treated as FATAL, because it can easily result in seriously malformed hashes,
            and we can't correct for it ourselves.

            test cases from <http://cvsweb.openwall.com/cgi/cvsweb.cgi/Owl/packages/glibc/crypt_blowfish/wrapper.c.diff?r1=1.9;r2=1.10>
            reference hash is the incorrectly generated $2x$ hash taken from above url
            """
        secret = b'\xd1\x91'
        bug_hash = ident.encode('ascii') + b'05$6bNw2HLQYeqHYyBfLMsv/OiwqTymGIGzFsA4hOTWebfehXHNprcAS'
        correct_hash = ident.encode('ascii') + b'05$6bNw2HLQYeqHYyBfLMsv/OUcZd0LKP39b87nBw3.S2tVZSqiQX6eu'
        if verify(secret, bug_hash):
            raise PasslibSecurityError('passlib.hash.bcrypt: Your installation of the %r backend is vulnerable to the crypt_blowfish 8-bit bug (CVE-2011-2483) under %r hashes, and should be upgraded or replaced with another backend' % (backend, ident))
        if not verify(secret, correct_hash):
            raise RuntimeError('%s backend failed to verify %s 8bit hash' % (backend, ident))

    def detect_wrap_bug(ident):
        """
            check for bsd wraparound bug (fixed in 2b)
            this is treated as a warning, because it's rare in the field,
            and pybcrypt (as of 2015-7-21) is unpatched, but some people may be stuck with it.

            test cases from <http://www.openwall.com/lists/oss-security/2012/01/02/4>

            NOTE: reference hash is of password "0"*72

            NOTE: if in future we need to deliberately create hashes which have this bug,
                  can use something like 'hashpw(repeat_string(secret[:((1+secret) % 256) or 1]), 72)'
            """
        secret = (b'0123456789' * 26)[:255]
        bug_hash = ident.encode('ascii') + b'04$R1lJ2gkNaoPGdafE.H.16.nVyh2niHsGJhayOHLMiXlI45o8/DU.6'
        if verify(secret, bug_hash):
            return True
        correct_hash = ident.encode('ascii') + b'04$R1lJ2gkNaoPGdafE.H.16.1MKHPvmKwryeulRe225LKProWYwt9Oi'
        if not verify(secret, correct_hash):
            raise RuntimeError('%s backend failed to verify %s wraparound hash' % (backend, ident))
        return False

    def assert_lacks_wrap_bug(ident):
        if not detect_wrap_bug(ident):
            return
        raise RuntimeError('%s backend unexpectedly has wraparound bug for %s' % (backend, ident))
    test_hash_20 = b'$2$04$5BJqKfqMQvV7nS.yUguNcuRfMMOXK0xPWavM7pOzjEi5ze5T1k8/S'
    result = safe_verify('test', test_hash_20)
    if result is NotImplemented:
        mixin_cls._lacks_20_support = True
        log.debug('%r backend lacks $2$ support, enabling workaround', backend)
    elif not result:
        raise RuntimeError('%s incorrectly rejected $2$ hash' % backend)
    result = safe_verify('test', TEST_HASH_2A)
    if result is NotImplemented:
        raise RuntimeError('%s lacks support for $2a$ hashes' % backend)
    elif not result:
        raise RuntimeError('%s incorrectly rejected $2a$ hash' % backend)
    else:
        assert_lacks_8bit_bug(IDENT_2A)
        if detect_wrap_bug(IDENT_2A):
            if backend == 'os_crypt':
                log.debug('%r backend has $2a$ bsd wraparound bug, enabling workaround', backend)
            else:
                warn('passlib.hash.bcrypt: Your installation of the %r backend is vulnerable to the bsd wraparound bug, and should be upgraded or replaced with another backend (enabling workaround for now).' % backend, uh.exc.PasslibSecurityWarning)
            mixin_cls._has_2a_wraparound_bug = True
    test_hash_2y = TEST_HASH_2A.replace('2a', '2y')
    result = safe_verify('test', test_hash_2y)
    if result is NotImplemented:
        mixin_cls._lacks_2y_support = True
        log.debug('%r backend lacks $2y$ support, enabling workaround', backend)
    elif not result:
        raise RuntimeError('%s incorrectly rejected $2y$ hash' % backend)
    else:
        assert_lacks_8bit_bug(IDENT_2Y)
        assert_lacks_wrap_bug(IDENT_2Y)
    test_hash_2b = TEST_HASH_2A.replace('2a', '2b')
    result = safe_verify('test', test_hash_2b)
    if result is NotImplemented:
        mixin_cls._lacks_2b_support = True
        log.debug('%r backend lacks $2b$ support, enabling workaround', backend)
    elif not result:
        raise RuntimeError('%s incorrectly rejected $2b$ hash' % backend)
    else:
        mixin_cls._fallback_ident = IDENT_2B
        assert_lacks_8bit_bug(IDENT_2B)
        assert_lacks_wrap_bug(IDENT_2B)
    mixin_cls._workrounds_initialized = True
    return True