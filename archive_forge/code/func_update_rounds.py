from __future__ import absolute_import, division, print_function
import logging; log = logging.getLogger(__name__)
from passlib.utils.compat import suppress_cause
from passlib.ext.django.utils import DJANGO_VERSION, DjangoTranslator, _PasslibHasherWrapper
from passlib.tests.utils import TestCase, TEST_MODE
from .test_ext_django import (
def update_rounds():
    """
                sync django hasher config -> passlib hashers
                """
    for handler in context.schemes(resolve=True):
        if 'rounds' not in handler.setting_kwds:
            continue
        hasher = adapter.passlib_to_django(handler)
        if isinstance(hasher, _PasslibHasherWrapper):
            continue
        rounds = getattr(hasher, 'rounds', None) or getattr(hasher, 'iterations', None)
        if rounds is None:
            continue
        handler.min_desired_rounds = handler.max_desired_rounds = handler.default_rounds = rounds