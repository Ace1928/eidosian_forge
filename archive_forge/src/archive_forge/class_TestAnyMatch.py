from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._higherorder import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestAnyMatch(TestCase, TestMatchersInterface):
    matches_matcher = AnyMatch(Equals('elephant'))
    matches_matches = [['grass', 'cow', 'steak', 'milk', 'elephant'], (13, 'elephant'), ['elephant', 'elephant', 'elephant'], {'hippo', 'rhino', 'elephant'}]
    matches_mismatches = [[], ['grass', 'cow', 'steak', 'milk'], (13, 12, 10), ['element', 'hephalump', 'pachyderm'], {'hippo', 'rhino', 'diplodocus'}]
    str_examples = [("AnyMatch(Equals('elephant'))", AnyMatch(Equals('elephant')))]
    describe_examples = [('Differences: [\n11 != 7\n9 != 7\n10 != 7\n]', [11, 9, 10], AnyMatch(Equals(7)))]