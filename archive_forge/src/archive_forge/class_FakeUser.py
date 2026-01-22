from __future__ import absolute_import, division, print_function
import logging; log = logging.getLogger(__name__)
import sys
import re
from passlib import apps as _apps, exc, registry
from passlib.apps import django10_context, django14_context, django16_context
from passlib.context import CryptContext
from passlib.ext.django.utils import (
from passlib.utils.compat import iteritems, get_method_function, u
from passlib.utils.decor import memoized_property
from passlib.tests.utils import TestCase, TEST_MODE, handler_derived_from
from passlib.tests.test_handlers import get_handler_case
from passlib.hash import django_pbkdf2_sha256
class FakeUser(User):
    """mock user object for use in testing"""

    class Meta:
        app_label = __name__

    @memoized_property
    def saved_passwords(self):
        return []

    def pop_saved_passwords(self):
        try:
            return self.saved_passwords[:]
        finally:
            del self.saved_passwords[:]

    def save(self, update_fields=None):
        self.saved_passwords.append(self.password)