from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
class TestOnException(TestCase):
    run_test_with = FullStackRunTest

    def test_default_works(self):
        events = []

        class Case(TestCase):

            def method(self):
                self.onException(an_exc_info)
                events.append(True)
        case = Case('method')
        case.run()
        self.assertThat(events, Equals([True]))

    def test_added_handler_works(self):
        events = []

        class Case(TestCase):

            def method(self):
                self.addOnException(events.append)
                self.onException(an_exc_info)
        case = Case('method')
        case.run()
        self.assertThat(events, Equals([an_exc_info]))

    def test_handler_that_raises_is_not_caught(self):
        events = []

        class Case(TestCase):

            def method(self):
                self.addOnException(events.index)
                self.assertThat(lambda: self.onException(an_exc_info), Raises(MatchesException(ValueError)))
        case = Case('method')
        case.run()
        self.assertThat(events, Equals([]))