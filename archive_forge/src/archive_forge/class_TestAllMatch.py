from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._higherorder import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestAllMatch(TestCase, TestMatchersInterface):
    matches_matcher = AllMatch(LessThan(10))
    matches_matches = [[9, 9, 9], (9, 9), iter([9, 9, 9, 9, 9])]
    matches_mismatches = [[11, 9, 9], iter([9, 12, 9, 11])]
    str_examples = [('AllMatch(LessThan(12))', AllMatch(LessThan(12)))]
    describe_examples = [('Differences: [\n11 >= 10\n10 >= 10\n]', [11, 9, 10], AllMatch(LessThan(10)))]