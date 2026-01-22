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
class ExtensionBehaviorTest(DjangoBehaviorTest):
    """
    test that "passlib.ext.django" conforms to behavioral assertions in DjangoBehaviorTest
    """
    descriptionPrefix = 'verify extension behavior'
    config = dict(schemes='sha256_crypt,md5_crypt,des_crypt', deprecated='des_crypt')

    def setUp(self):
        super(ExtensionBehaviorTest, self).setUp()
        self.load_extension(PASSLIB_CONFIG=self.config)
        self.patched = True