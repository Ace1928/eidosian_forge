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
def TEST_MODE(min=None, max=None):
    """check if test for specified mode should be enabled.

    ``"quick"``
        run the bare minimum tests to ensure functionality.
        variable-cost hashes are tested at their lowest setting.
        hash algorithms are only tested against the backend that will
        be used on the current host. no fuzz testing is done.

    ``"default"``
        same as ``"quick"``, except: hash algorithms are tested
        at default levels, and a brief round of fuzz testing is done
        for each hash.

    ``"full"``
        extra regression and internal tests are enabled, hash algorithms are tested
        against all available backends, unavailable ones are mocked whre possible,
        additional time is devoted to fuzz testing.
    """
    if min and _test_mode < _TEST_MODES.index(min):
        return False
    if max and _test_mode > _TEST_MODES.index(max):
        return False
    return True