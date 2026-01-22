from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._higherorder import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestNotInterface(TestCase, TestMatchersInterface):
    matches_matcher = Not(Equals(1))
    matches_matches = [2]
    matches_mismatches = [1]
    str_examples = [('Not(Equals(1))', Not(Equals(1))), ("Not(Equals('1'))", Not(Equals('1')))]
    describe_examples = [('1 matches Equals(1)', 1, Not(Equals(1)))]