from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class _ldap_sha1_crypt_test(HandlerCase):
    handler = hash.ldap_sha1_crypt
    known_correct_hashes = [('password', '{CRYPT}$sha1$10$c.mcTzCw$gF8UeYst9yXX7WNZKc5Fjkq0.au7'), (UPASS_TABLE, '{CRYPT}$sha1$10$rnqXlOsF$aGJf.cdRPewJAXo1Rn1BkbaYh0fP')]

    def populate_settings(self, kwds):
        kwds.setdefault('rounds', 10)
        super(_ldap_sha1_crypt_test, self).populate_settings(kwds)

    def test_77_fuzz_input(self, **ignored):
        raise self.skipTest('unneeded')