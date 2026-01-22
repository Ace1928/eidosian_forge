import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
class TestExtendedToOriginalResultDecoratorBase(TestCase):

    def make_26_result(self):
        self.result = Python26TestResult()
        self.make_converter()

    def make_27_result(self):
        self.result = Python27TestResult()
        self.make_converter()

    def make_converter(self):
        self.converter = ExtendedToOriginalDecorator(self.result)

    def make_extended_result(self):
        self.result = ExtendedTestResult()
        self.make_converter()

    def check_outcome_details(self, outcome):
        """Call an outcome with a details dict to be passed through."""
        details = {'foo': 'bar'}
        getattr(self.converter, outcome)(self, details=details)
        self.assertEqual([(outcome, self, details)], self.result._events)

    def get_details_and_string(self):
        """Get a details dict and expected string."""
        text1 = lambda: [_b('1\n2\n')]
        text2 = lambda: [_b('3\n4\n')]
        bin1 = lambda: [_b('5\n')]
        details = {'text 1': Content(ContentType('text', 'plain'), text1), 'text 2': Content(ContentType('text', 'strange'), text2), 'bin 1': Content(ContentType('application', 'binary'), bin1)}
        return (details, 'Binary content:\n  bin 1 (application/binary)\n\ntext 1: {{{\n1\n2\n}}}\n\ntext 2: {{{\n3\n4\n}}}\n')

    def check_outcome_details_to_exec_info(self, outcome, expected=None):
        """Call an outcome with a details dict to be made into exc_info."""
        if not expected:
            expected = outcome
        details, err_str = self.get_details_and_string()
        getattr(self.converter, outcome)(self, details=details)
        err = self.converter._details_to_exc_info(details)
        self.assertEqual([(expected, self, err)], self.result._events)

    def check_outcome_details_to_nothing(self, outcome, expected=None):
        """Call an outcome with a details dict to be swallowed."""
        if not expected:
            expected = outcome
        details = {'foo': 'bar'}
        getattr(self.converter, outcome)(self, details=details)
        self.assertEqual([(expected, self)], self.result._events)

    def check_outcome_details_to_string(self, outcome):
        """Call an outcome with a details dict to be stringified."""
        details, err_str = self.get_details_and_string()
        getattr(self.converter, outcome)(self, details=details)
        self.assertEqual([(outcome, self, err_str)], self.result._events)

    def check_outcome_details_to_arg(self, outcome, arg, extra_detail=None):
        """Call an outcome with a details dict to have an arg extracted."""
        details, _ = self.get_details_and_string()
        if extra_detail:
            details.update(extra_detail)
        getattr(self.converter, outcome)(self, details=details)
        self.assertEqual([(outcome, self, arg)], self.result._events)

    def check_outcome_exc_info(self, outcome, expected=None):
        """Check that calling a legacy outcome still works."""
        if not expected:
            expected = outcome
        err = sys.exc_info()
        getattr(self.converter, outcome)(self, err)
        self.assertEqual([(expected, self, err)], self.result._events)

    def check_outcome_exc_info_to_nothing(self, outcome, expected=None):
        """Check that calling a legacy outcome on a fallback works."""
        if not expected:
            expected = outcome
        err = sys.exc_info()
        getattr(self.converter, outcome)(self, err)
        self.assertEqual([(expected, self)], self.result._events)

    def check_outcome_nothing(self, outcome, expected=None):
        """Check that calling a legacy outcome still works."""
        if not expected:
            expected = outcome
        getattr(self.converter, outcome)(self)
        self.assertEqual([(expected, self)], self.result._events)

    def check_outcome_string_nothing(self, outcome, expected):
        """Check that calling outcome with a string calls expected."""
        getattr(self.converter, outcome)(self, 'foo')
        self.assertEqual([(expected, self)], self.result._events)

    def check_outcome_string(self, outcome):
        """Check that calling outcome with a string works."""
        getattr(self.converter, outcome)(self, 'foo')
        self.assertEqual([(outcome, self, 'foo')], self.result._events)