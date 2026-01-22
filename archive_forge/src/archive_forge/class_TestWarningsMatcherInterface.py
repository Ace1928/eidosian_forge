import warnings
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._warnings import Warnings, IsDeprecated, WarningMessage
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestWarningsMatcherInterface(TestCase, TestMatchersInterface):
    """
    Tests for `testtools.matchers._warnings.Warnings`.

    Specifically with the optional matcher argument.
    """
    matches_matcher = Warnings(warnings_matcher=MatchesListwise([MatchesStructure(message=AfterPreprocessing(str, Contains('old_func')))]))

    def old_func():
        warnings.warn('old_func is deprecated', DeprecationWarning, 2)

    def older_func():
        warnings.warn('older_func is deprecated', DeprecationWarning, 2)
    matches_matches = [old_func]
    matches_mismatches = [lambda: None, older_func]
    str_examples = []
    describe_examples = []