from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class htdigest_test(UserHandlerMixin, HandlerCase):
    handler = hash.htdigest
    known_correct_hashes = [(('Circle Of Life', 'Mufasa', 'testrealm@host.com'), '939e7578ed9e3c518a452acee763bce9'), ((UPASS_TABLE, UPASS_USD, UPASS_WAV), '4dabed2727d583178777fab468dd1f17')]
    known_unidentified_hashes = ['939e7578edAe3c518a452acee763bce9', '939e7578edxe3c518a452acee763bce9']

    def test_80_user(self):
        raise self.skipTest("test case doesn't support 'realm' keyword")

    def populate_context(self, secret, kwds):
        """insert username into kwds"""
        if isinstance(secret, tuple):
            secret, user, realm = secret
        else:
            user, realm = ('user', 'realm')
        kwds.setdefault('user', user)
        kwds.setdefault('realm', realm)
        return secret