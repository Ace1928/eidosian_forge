import warnings
from typing import Optional, TypeVar, cast, overload
from hamcrest.core.matcher import Matcher
from hamcrest.core.string_description import StringDescription
Asserts that actual value satisfies matcher. (Can also assert plain
    boolean condition.)

    :param actual: The object to evaluate as the actual value.
    :param matcher: The matcher to satisfy as the expected condition.
    :param reason: Optional explanation to include in failure description.

    ``assert_that`` passes the actual value to the matcher for evaluation. If
    the matcher is not satisfied, an exception is thrown describing the
    mismatch.

    ``assert_that`` is designed to integrate well with PyUnit and other unit
    testing frameworks. The exception raised for an unmet assertion is an
    :py:exc:`AssertionError`, which PyUnit reports as a test failure.

    With a different set of parameters, ``assert_that`` can also verify a
    boolean condition:

    .. function:: assert_that(assertion[, reason])

    :param assertion:  Boolean condition to verify.
    :param reason:  Optional explanation to include in failure description.

    This is equivalent to the :py:meth:`~unittest.TestCase.assertTrue` method
    of :py:class:`unittest.TestCase`, but offers greater flexibility in test
    writing by being a standalone function.

    