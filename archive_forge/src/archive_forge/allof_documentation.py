from typing import Optional, TypeVar, Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
Matches if all of the given matchers evaluate to ``True``.

    :param matcher1,...:  A comma-separated list of matchers.

    The matchers are evaluated from left to right using short-circuit
    evaluation, so evaluation stops as soon as a matcher returns ``False``.

    Any argument that is not a matcher is implicitly wrapped in an
    :py:func:`~hamcrest.core.core.isequal.equal_to` matcher to check for
    equality.

    