from typing import Type
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import is_matchable_type
from hamcrest.core.matcher import Matcher
Matches if object is an instance of, or inherits from, a given type.

    :param atype: The type to compare against as the expected type.

    This matcher checks whether the evaluated object is an instance of
    ``atype`` or an instance of any class that inherits from ``atype``.

    Example::

        instance_of(str)

    