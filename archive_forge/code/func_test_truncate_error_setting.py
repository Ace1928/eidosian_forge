from __future__ import with_statement
from binascii import unhexlify
import contextlib
from functools import wraps, partial
import hashlib
import logging; log = logging.getLogger(__name__)
import random
import re
import os
import sys
import tempfile
import threading
import time
from passlib.exc import PasslibHashWarning, PasslibConfigWarning
from passlib.utils.compat import PY3, JYTHON
import warnings
from warnings import warn
from passlib import exc
from passlib.exc import MissingBackendError
import passlib.registry as registry
from passlib.tests.backports import TestCase as _TestCase, skip, skipIf, skipUnless, SkipTest
from passlib.utils import has_rounds_info, has_salt_info, rounds_cost_values, \
from passlib.utils.compat import iteritems, irange, u, unicode, PY2, nullcontext
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
def test_truncate_error_setting(self):
    """
        validate 'truncate_error' setting & related attributes
        """
    hasher = self.handler
    if hasher.truncate_size is None:
        self.assertNotIn('truncate_error', hasher.setting_kwds)
        return
    if not hasher.truncate_error:
        self.assertFalse(hasher.truncate_verify_reject)
    if 'truncate_error' not in hasher.setting_kwds:
        self.assertTrue(hasher.truncate_error)
        return

    def parse_value(value):
        return hasher.using(truncate_error=value).truncate_error
    self.assertEqual(parse_value(None), hasher.truncate_error)
    self.assertEqual(parse_value(True), True)
    self.assertEqual(parse_value('true'), True)
    self.assertEqual(parse_value(False), False)
    self.assertEqual(parse_value('false'), False)
    self.assertRaises(ValueError, parse_value, 'xxx')