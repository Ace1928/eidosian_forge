from typing import Optional, Type, TypeVar, overload, Any
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import is_matchable_type, wrap_matcher
from hamcrest.core.matcher import Matcher
from .isinstanceof import instance_of
Decorates another matcher, or provides shortcuts to the frequently used
    ``is(equal_to(x))`` and ``is(instance_of(x))``.

    :param x: The matcher to satisfy,  or a type for
        :py:func:`~hamcrest.core.core.isinstanceof.instance_of` matching, or an
        expected value for :py:func:`~hamcrest.core.core.isequal.equal_to`
        matching.

    This matcher compares the evaluated object to the given matcher.

    .. note::

        PyHamcrest's ``is_`` matcher is unrelated to Python's ``is`` operator.
        The matcher for object identity is
        :py:func:`~hamcrest.core.core.issame.same_instance`.

    If the ``x`` argument is a matcher, its behavior is retained, but the test
    may be more expressive. For example::

        assert_that(value, less_than(5))
        assert_that(value, is_(less_than(5)))

    If the ``x`` argument is a type, it is wrapped in an
    :py:func:`~hamcrest.core.core.isinstanceof.instance_of` matcher. This makes
    the following statements equivalent::

        assert_that(cheese, instance_of(Cheddar))
        assert_that(cheese, is_(instance_of(Cheddar)))
        assert_that(cheese, is_(Cheddar))

    Otherwise, if the ``x`` argument is not a matcher, it is wrapped in an
    :py:func:`~hamcrest.core.core.isequal.equal_to` matcher. This makes the
    following statements equivalent::

        assert_that(cheese, equal_to(smelly))
        assert_that(cheese, is_(equal_to(smelly)))
        assert_that(cheese, is_(smelly))

    Choose the style that makes your expression most readable. This will vary
    depending on context.

    