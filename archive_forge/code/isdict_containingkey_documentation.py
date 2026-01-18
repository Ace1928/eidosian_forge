from typing import Any, Hashable, Mapping, TypeVar, Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
Matches if dictionary contains an entry whose key satisfies a given
    matcher.

    :param key_match: The matcher to satisfy for the key, or an expected value
        for :py:func:`~hamcrest.core.core.isequal.equal_to` matching.

    This matcher iterates the evaluated dictionary, searching for any key-value
    entry whose key satisfies the given matcher. If a matching entry is found,
    ``has_key`` is satisfied.

    Any argument that is not a matcher is implicitly wrapped in an
    :py:func:`~hamcrest.core.core.isequal.equal_to` matcher to check for
    equality.

    Examples::

        has_key(equal_to('foo'))
        has_key('foo')

    