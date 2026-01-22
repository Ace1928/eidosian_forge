import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
class CheckVariableNameTests(TestCase):

    def test_invalid(self):
        self.assertFalse(_check_variable_name(b'foo '))
        self.assertFalse(_check_variable_name(b'bar,bar'))
        self.assertFalse(_check_variable_name(b'bar.bar'))

    def test_valid(self):
        self.assertTrue(_check_variable_name(b'FOO'))
        self.assertTrue(_check_variable_name(b'foo'))
        self.assertTrue(_check_variable_name(b'foo-bar'))