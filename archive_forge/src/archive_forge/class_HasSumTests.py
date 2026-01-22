from typing import Callable, Sequence, Tuple, Type
from hamcrest import anything, assert_that, contains, contains_string, equal_to, not_
from hamcrest.core.matcher import Matcher
from hamcrest.core.string_description import StringDescription
from hypothesis import given
from hypothesis.strategies import (
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .matchers import HasSum, IsSequenceOf, S, isFailure, similarFrame
class HasSumTests(SynchronousTestCase):
    """
    Tests for L{HasSum}.
    """
    summables = one_of(tuples(lists(integers()), just(concatInt)), tuples(lists(text()), just(concatStr)), tuples(lists(binary()), just(concatBytes)))

    @given(summables)
    def test_matches(self, summable: Tuple[Sequence[S], Summer[S]]) -> None:
        """
        L{HasSum} matches a sequence if the elements sum to a value matched by
        the parameterized matcher.

        :param summable: A tuple of a sequence of values to try to match and a
            function which can compute the correct sum for that sequence.
        """
        seq, sumFunc = summable
        expected = sumFunc(seq)
        zero = sumFunc([])
        matcher = HasSum(equal_to(expected), zero)
        description = StringDescription()
        assert_that(matcher.matches(seq, description), equal_to(True))
        assert_that(str(description), equal_to(''))

    @given(summables)
    def test_mismatches(self, summable: Tuple[Sequence[S], Summer[S]]) -> None:
        """
        L{HasSum} does not match a sequence if the elements do not sum to a
        value matched by the parameterized matcher.

        :param summable: See L{test_matches}.
        """
        seq, sumFunc = summable
        zero = sumFunc([])
        sumMatcher: Matcher[S] = not_(anything())
        matcher = HasSum(sumMatcher, zero)
        actualDescription = StringDescription()
        assert_that(matcher.matches(seq, actualDescription), equal_to(False))
        sumMatcherDescription = StringDescription()
        sumMatcherDescription.append_description_of(sumMatcher)
        actualStr = str(actualDescription)
        assert_that(actualStr, contains_string('a sequence with sum'))
        assert_that(actualStr, contains_string(str(sumMatcherDescription)))