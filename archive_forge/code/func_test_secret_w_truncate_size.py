from __future__ import with_statement
from binascii import unhexlify
import contextlib
from functools import wraps, partial
import hashlib
import logging; log = logging.getLogger(__name__)
import random
import re
import os
import sys
import tempfile
import threading
import time
from passlib.exc import PasslibHashWarning, PasslibConfigWarning
from passlib.utils.compat import PY3, JYTHON
import warnings
from warnings import warn
from passlib import exc
from passlib.exc import MissingBackendError
import passlib.registry as registry
from passlib.tests.backports import TestCase as _TestCase, skip, skipIf, skipUnless, SkipTest
from passlib.utils import has_rounds_info, has_salt_info, rounds_cost_values, \
from passlib.utils.compat import iteritems, irange, u, unicode, PY2, nullcontext
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
def test_secret_w_truncate_size(self):
    """
        test password size limits raise truncate_error (if appropriate)
        """
    handler = self.handler
    truncate_size = handler.truncate_size
    if not truncate_size:
        raise self.skipTest('truncate_size not set')
    size_error_type = exc.PasswordSizeError
    if 'truncate_error' in handler.setting_kwds:
        without_error = handler.using(truncate_error=False)
        with_error = handler.using(truncate_error=True)
        size_error_type = exc.PasswordTruncateError
    elif handler.truncate_error:
        without_error = None
        with_error = handler
    else:
        without_error = handler
        with_error = None
    base = 'too many secrets'
    alt = 'x'
    long_secret = repeat_string(base, truncate_size + 1)
    short_secret = long_secret[:-1]
    alt_long_secret = long_secret[:-1] + alt
    alt_short_secret = short_secret[:-1] + alt
    short_verify_success = not handler.is_disabled
    long_verify_success = short_verify_success and (not handler.truncate_verify_reject)
    assert without_error or with_error
    for cand_hasher in [without_error, with_error]:
        short_hash = self.do_encrypt(short_secret, handler=cand_hasher)
        self.assertEqual(self.do_verify(short_secret, short_hash, handler=cand_hasher), short_verify_success)
        self.assertFalse(self.do_verify(alt_short_secret, short_hash, handler=with_error), 'truncate_size value is too large')
        self.assertEqual(self.do_verify(long_secret, short_hash, handler=cand_hasher), long_verify_success)
    if without_error:
        long_hash = self.do_encrypt(long_secret, handler=without_error)
        self.assertEqual(self.do_verify(long_secret, long_hash, handler=without_error), short_verify_success)
        self.assertEqual(self.do_verify(alt_long_secret, long_hash, handler=without_error), short_verify_success)
        self.assertTrue(self.do_verify(short_secret, long_hash, handler=without_error))
    if with_error:
        err = self.assertRaises(size_error_type, self.do_encrypt, long_secret, handler=with_error)
        self.assertEqual(err.max_size, truncate_size)