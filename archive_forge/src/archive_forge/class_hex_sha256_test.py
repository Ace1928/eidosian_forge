from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class hex_sha256_test(HandlerCase):
    handler = hash.hex_sha256
    known_correct_hashes = [('password', '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8'), (UPASS_TABLE, '6ed729e19bf24d3d20f564375820819932029df05547116cfc2cc868a27b4493')]