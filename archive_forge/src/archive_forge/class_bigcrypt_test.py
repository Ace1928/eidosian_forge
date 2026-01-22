from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class bigcrypt_test(HandlerCase):
    handler = hash.bigcrypt
    known_correct_hashes = [('passphrase', 'qiyh4XPJGsOZ2MEAyLkfWqeQ'), ('This is very long passwd', 'f8.SVpL2fvwjkAnxn8/rgTkwvrif6bjYB5c'), (UPASS_TABLE, 'SEChBAyMbMNhgGLyP7kD1HZU')]
    known_unidentified_hashes = ['qiyh4XPJGsOZ2MEAyLkfWqef8.SVpL2fvwjkAnxn8/rgTkwvrif6bjYB5cd']
    known_other_hashes = [row for row in HandlerCase.known_other_hashes if row[0] != 'des_crypt']

    def test_90_internal(self):
        self.assertRaises(ValueError, hash.bigcrypt, use_defaults=True, checksum=u('yh4XPJGsOZ'))