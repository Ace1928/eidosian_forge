import warnings
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._warnings import Warnings, IsDeprecated, WarningMessage
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestWarningMessageLineInterface(TestCase, TestMatchersInterface):
    """
    Tests for `testtools.matchers._warnings.WarningMessage`.

    In particular matching the ``line``.
    """
    matches_matcher = WarningMessage(category_type=DeprecationWarning, line=Equals('x'))
    warning_foo = make_warning_message('foo', DeprecationWarning, line='x')
    warning_bar = make_warning_message('bar', DeprecationWarning, line='y')
    matches_matches = [warning_foo]
    matches_mismatches = [warning_bar]
    str_examples = []
    describe_examples = []