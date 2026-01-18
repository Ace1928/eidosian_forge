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
def test_using_version(self):
    handler = self.handler
    self.assertEqual(handler.version, 2)
    subcls = handler.using(version=1)
    self.assertEqual(subcls.version, 1)
    self.assertRaises(ValueError, handler.using, version=999)
    subcls = handler.using(version=1, ident='2a')
    self.assertRaises(ValueError, handler.using, ident='2a')