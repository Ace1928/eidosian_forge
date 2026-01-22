from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._higherorder import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchesAllInterface(TestCase, TestMatchersInterface):
    matches_matcher = MatchesAll(NotEquals(1), NotEquals(2))
    matches_matches = [3, 4]
    matches_mismatches = [1, 2]
    str_examples = [('MatchesAll(NotEquals(1), NotEquals(2))', MatchesAll(NotEquals(1), NotEquals(2)))]
    describe_examples = [('Differences: [\n1 == 1\n]', 1, MatchesAll(NotEquals(1), NotEquals(2))), ('1 == 1', 1, MatchesAll(NotEquals(2), NotEquals(1), Equals(3), first_only=True))]