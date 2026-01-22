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
class UnsaltedHash(uh.StaticHandler):
    """test algorithm which lacks a salt"""
    name = 'unsalted_test_hash'
    checksum_chars = uh.LOWER_HEX_CHARS
    checksum_size = 40

    def _calc_checksum(self, secret):
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        data = b'boblious' + secret
        return str_to_uascii(hashlib.sha1(data).hexdigest())