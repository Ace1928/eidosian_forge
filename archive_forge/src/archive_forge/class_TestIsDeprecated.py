import warnings
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._warnings import Warnings, IsDeprecated, WarningMessage
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestIsDeprecated(TestCase):
    """
    Tests for `testtools.matchers._warnings.IsDeprecated`.
    """
    run_tests_with = FullStackRunTest

    def test_warning(self):

        def old_func():
            warnings.warn('old_func is deprecated', DeprecationWarning, 2)
        self.assertThat(old_func, IsDeprecated(Contains('old_func')))