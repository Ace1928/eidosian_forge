from typing import Type, TypeVar, Union, overload
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import is_matchable_type, wrap_matcher
from hamcrest.core.matcher import Matcher
from .isinstanceof import instance_of
Alias of :py:func:`is_not` for better readability of negations.

    Examples::

        assert_that(alist, not_(has_item(item)))

    