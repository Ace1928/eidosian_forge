from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import re
import warnings
from passlib import hash
from passlib.utils import repeat_string
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, SkipTest
from passlib.tests.test_handlers import UPASS_USD, UPASS_TABLE
from passlib.tests.test_ext_django import DJANGO_VERSION, MIN_DJANGO_VERSION, \
from passlib.tests.test_handlers_argon2 import _base_argon2_test
def _require_django_support(self):
    if DJANGO_VERSION < self.min_django_version:
        raise self.skipTest('Django >= %s not installed' % vstr(self.min_django_version))
    if self.max_django_version and DJANGO_VERSION > self.max_django_version:
        raise self.skipTest('Django <= %s not installed' % vstr(self.max_django_version))
    name = self.handler.django_name
    if not check_django_hasher_has_backend(name):
        raise self.skipTest('django hasher %r not available' % name)
    return True