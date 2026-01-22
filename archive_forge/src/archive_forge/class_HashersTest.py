from __future__ import absolute_import, division, print_function
import logging; log = logging.getLogger(__name__)
from passlib.utils.compat import suppress_cause
from passlib.ext.django.utils import DJANGO_VERSION, DjangoTranslator, _PasslibHasherWrapper
from passlib.tests.utils import TestCase, TEST_MODE
from .test_ext_django import (
class HashersTest(TestCase):

    def test_external_django_hasher_tests(self):
        """external django hasher tests"""
        raise self.skipTest(hashers_skip_msg)