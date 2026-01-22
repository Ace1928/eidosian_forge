from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class hex_sha512_test(HandlerCase):
    handler = hash.hex_sha512
    known_correct_hashes = [('password', 'b109f3bbbc244eb82441917ed06d618b9008dd09b3befd1b5e07394c706a8bb980b1d7785e5976ec049b46df5f1326af5a2ea6d103fd07c95385ffab0cacbc86'), (UPASS_TABLE, 'd91bb0a23d66dca07a1781fd63ae6a05f6919ee5fc368049f350c9f293b078a18165d66097cf0d89fdfbeed1ad6e7dba2344e57348cd6d51308c843a06f29caf')]