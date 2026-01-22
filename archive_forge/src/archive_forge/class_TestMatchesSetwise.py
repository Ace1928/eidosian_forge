import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchesSetwise(TestCase):
    run_tests_with = FullStackRunTest

    def assertMismatchWithDescriptionMatching(self, value, matcher, description_matcher):
        mismatch = matcher.match(value)
        if mismatch is None:
            self.fail(f'{matcher} matched {value}')
        actual_description = mismatch.describe()
        self.assertThat(actual_description, Annotate(f'{matcher} matching {value}', description_matcher))

    def test_matches(self):
        self.assertIs(None, MatchesSetwise(Equals(1), Equals(2)).match([2, 1]))

    def test_mismatches(self):
        self.assertMismatchWithDescriptionMatching([2, 3], MatchesSetwise(Equals(1), Equals(2)), MatchesRegex('.*There was 1 mismatch$', re.S))

    def test_too_many_matchers(self):
        self.assertMismatchWithDescriptionMatching([2, 3], MatchesSetwise(Equals(1), Equals(2), Equals(3)), Equals('There was 1 matcher left over: Equals(1)'))

    def test_too_many_values(self):
        self.assertMismatchWithDescriptionMatching([1, 2, 3], MatchesSetwise(Equals(1), Equals(2)), Equals('There was 1 value left over: [3]'))

    def test_two_too_many_matchers(self):
        self.assertMismatchWithDescriptionMatching([3], MatchesSetwise(Equals(1), Equals(2), Equals(3)), MatchesRegex('There were 2 matchers left over: Equals\\([12]\\), Equals\\([12]\\)'))

    def test_two_too_many_values(self):
        self.assertMismatchWithDescriptionMatching([1, 2, 3, 4], MatchesSetwise(Equals(1), Equals(2)), MatchesRegex('There were 2 values left over: \\[[34], [34]\\]'))

    def test_mismatch_and_too_many_matchers(self):
        self.assertMismatchWithDescriptionMatching([2, 3], MatchesSetwise(Equals(0), Equals(1), Equals(2)), MatchesRegex('.*There was 1 mismatch and 1 extra matcher: Equals\\([01]\\)', re.S))

    def test_mismatch_and_too_many_values(self):
        self.assertMismatchWithDescriptionMatching([2, 3, 4], MatchesSetwise(Equals(1), Equals(2)), MatchesRegex('.*There was 1 mismatch and 1 extra value: \\[[34]\\]', re.S))

    def test_mismatch_and_two_too_many_matchers(self):
        self.assertMismatchWithDescriptionMatching([3, 4], MatchesSetwise(Equals(0), Equals(1), Equals(2), Equals(3)), MatchesRegex('.*There was 1 mismatch and 2 extra matchers: Equals\\([012]\\), Equals\\([012]\\)', re.S))

    def test_mismatch_and_two_too_many_values(self):
        self.assertMismatchWithDescriptionMatching([2, 3, 4, 5], MatchesSetwise(Equals(1), Equals(2)), MatchesRegex('.*There was 1 mismatch and 2 extra values: \\[[145], [145]\\]', re.S))