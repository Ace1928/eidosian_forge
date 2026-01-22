import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._exception import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestRaisesExceptionMatcherInterface(TestCase, TestMatchersInterface):
    matches_matcher = Raises(exception_matcher=MatchesException(Exception('foo')))

    def boom_bar():
        raise Exception('bar')

    def boom_foo():
        raise Exception('foo')
    matches_matches = [boom_foo]
    matches_mismatches = [lambda: None, boom_bar]
    str_examples = []
    describe_examples = []