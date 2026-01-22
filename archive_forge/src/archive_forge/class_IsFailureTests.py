from typing import Callable, Sequence, Tuple, Type
from hamcrest import anything, assert_that, contains, contains_string, equal_to, not_
from hamcrest.core.matcher import Matcher
from hamcrest.core.string_description import StringDescription
from hypothesis import given
from hypothesis.strategies import (
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .matchers import HasSum, IsSequenceOf, S, isFailure, similarFrame
class IsFailureTests(SynchronousTestCase):
    """
    Tests for L{isFailure}.
    """

    @given(sampled_from([ValueError, ZeroDivisionError, RuntimeError]))
    def test_matches(self, excType: Type[BaseException]) -> None:
        """
        L{isFailure} matches instances of L{Failure} with matching
        attributes.

        :param excType: An exception type to wrap in a L{Failure} to be
            matched against.
        """
        matcher = isFailure(type=equal_to(excType))
        failure = Failure(excType())
        assert_that(matcher.matches(failure), equal_to(True))

    @given(sampled_from([ValueError, ZeroDivisionError, RuntimeError]))
    def test_mismatches(self, excType: Type[BaseException]) -> None:
        """
        L{isFailure} does not match instances of L{Failure} with
        attributes that don't match.

        :param excType: An exception type to wrap in a L{Failure} to be
            matched against.
        """
        matcher = isFailure(type=equal_to(excType), other=not_(anything()))
        failure = Failure(excType())
        assert_that(matcher.matches(failure), equal_to(False))

    def test_frames(self):
        """
        The L{similarFrame} matcher matches elements of the C{frames} list
        of a L{Failure}.
        """
        try:
            raise ValueError('Oh no')
        except BaseException:
            f = Failure()
        actualDescription = StringDescription()
        matcher = isFailure(frames=contains(similarFrame('test_frames', 'test_matchers')))
        assert_that(matcher.matches(f, actualDescription), equal_to(True), actualDescription)