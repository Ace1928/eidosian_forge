from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
def test_pairs(self):
    self._test_pair(hash.ldap_hex_sha1, 'sekrit', '{SHA}8d42e738c7adee551324955458b5e2c0b49ee655')
    self._test_pair(hash.ldap_hex_md5, 'sekrit', '{MD5}ccbc53f4464604e714f69dd11138d8b5')
    self._test_pair(hash.ldap_des_crypt, 'sekrit', '{CRYPT}nFia0rj2TT59A')
    self._test_pair(hash.roundup_plaintext, 'sekrit', '{plaintext}sekrit')
    self._test_pair(hash.ldap_pbkdf2_sha1, 'sekrit', '{PBKDF2}5000$7BvbBq.EZzz/O0HuwX3iP.nAG3s$g3oPnFFaga2BJaX5PoPRljl4XIE')