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
def _modify_django_config(kwds, sha_rounds=None):
    """
    helper to build django CryptContext config matching expected setup for stock django deploy.
    :param kwds:
    :param sha_rounds:
    :return:
    """
    if hasattr(kwds, 'to_dict'):
        kwds = kwds.to_dict()
    kwds.update(deprecated='auto')
    if sha_rounds is None and has_min_django:
        from django.contrib.auth.hashers import PBKDF2PasswordHasher
        sha_rounds = PBKDF2PasswordHasher.iterations
    if sha_rounds:
        kwds.update(django_pbkdf2_sha1__default_rounds=sha_rounds, django_pbkdf2_sha256__default_rounds=sha_rounds)
    return kwds