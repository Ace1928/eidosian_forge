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
def _create_using_rounds_helper(self):
    """
        setup test helpers for testing handler.using()'s rounds parameters.
        """
    self.require_rounds_info()
    handler = self.handler
    if handler.name == 'bsdi_crypt':
        orig_handler = handler
        handler = handler.using()
        handler._generate_rounds = classmethod(lambda cls: super(orig_handler, cls)._generate_rounds())
    orig_min_rounds = handler.min_rounds
    orig_max_rounds = handler.max_rounds
    orig_default_rounds = handler.default_rounds
    medium = ((orig_max_rounds or 9999) + orig_min_rounds) // 2
    if medium == orig_default_rounds:
        medium += 1
    small = (orig_min_rounds + medium) // 2
    large = ((orig_max_rounds or 9999) + medium) // 2
    if handler.name == 'bsdi_crypt':
        small |= 1
        medium |= 1
        large |= 1
        adj = 2
    else:
        adj = 1
    with self.assertWarningList([]):
        subcls = handler.using(min_desired_rounds=small, max_desired_rounds=large, default_rounds=medium)
    return (handler, subcls, small, medium, large, adj)