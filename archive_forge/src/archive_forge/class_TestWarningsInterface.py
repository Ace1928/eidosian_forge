import warnings
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._warnings import Warnings, IsDeprecated, WarningMessage
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestWarningsInterface(TestCase, TestMatchersInterface):
    """
    Tests for `testtools.matchers._warnings.Warnings`.

    Specifically without the optional argument.
    """
    matches_matcher = Warnings()

    def old_func():
        warnings.warn('old_func is deprecated', DeprecationWarning, 2)
    matches_matches = [old_func]
    matches_mismatches = [lambda: None]
    str_examples = []
    describe_examples = []