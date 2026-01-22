from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.handlers.bcrypt import IDENT_2, IDENT_2X
from passlib.utils import repeat_string, to_bytes, is_safe_crypt_input
from passlib.utils.compat import irange, PY3
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE
class bcrypt_os_crypt_test(_bcrypt_test.create_backend_case('os_crypt')):
    known_correct_hashes = [row for row in _bcrypt_test.known_correct_hashes if is_safe_crypt_input(row[0])]
    has_os_crypt_fallback = False