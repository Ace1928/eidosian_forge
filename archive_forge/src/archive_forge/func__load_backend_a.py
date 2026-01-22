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
@classmethod
def _load_backend_a(cls):
    if cls._enable_a:
        cls._set_calc_checksum_backend(cls._calc_checksum_a)
        return True
    else:
        return False