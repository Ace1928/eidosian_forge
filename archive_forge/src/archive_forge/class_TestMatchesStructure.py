import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchesStructure(TestCase, TestMatchersInterface):

    class SimpleClass:

        def __init__(self, x, y):
            self.x = x
            self.y = y
    matches_matcher = MatchesStructure(x=Equals(1), y=Equals(2))
    matches_matches = [SimpleClass(1, 2)]
    matches_mismatches = [SimpleClass(2, 2), SimpleClass(1, 1), SimpleClass(3, 3)]
    str_examples = [('MatchesStructure(x=Equals(1))', MatchesStructure(x=Equals(1))), ('MatchesStructure(y=Equals(2))', MatchesStructure(y=Equals(2))), ('MatchesStructure(x=Equals(1), y=Equals(2))', MatchesStructure(x=Equals(1), y=Equals(2)))]
    describe_examples = [('Differences: [\n1 != 3: x\n]', SimpleClass(1, 2), MatchesStructure(x=Equals(3), y=Equals(2))), ('Differences: [\n2 != 3: y\n]', SimpleClass(1, 2), MatchesStructure(x=Equals(1), y=Equals(3))), ('Differences: [\n1 != 0: x\n2 != 0: y\n]', SimpleClass(1, 2), MatchesStructure(x=Equals(0), y=Equals(0)))]

    def test_fromExample(self):
        self.assertThat(self.SimpleClass(1, 2), MatchesStructure.fromExample(self.SimpleClass(1, 3), 'x'))

    def test_byEquality(self):
        self.assertThat(self.SimpleClass(1, 2), MatchesStructure.byEquality(x=1))

    def test_withStructure(self):
        self.assertThat(self.SimpleClass(1, 2), MatchesStructure.byMatcher(LessThan, x=2))

    def test_update(self):
        self.assertThat(self.SimpleClass(1, 2), MatchesStructure(x=NotEquals(1)).update(x=Equals(1)))

    def test_update_none(self):
        self.assertThat(self.SimpleClass(1, 2), MatchesStructure(x=Equals(1), z=NotEquals(42)).update(z=None))