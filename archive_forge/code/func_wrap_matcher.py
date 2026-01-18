from typing import Type, TypeVar, Union
from hamcrest.core.base_matcher import Matcher
from hamcrest.core.core.isequal import equal_to
def wrap_matcher(x: Union[Matcher[T], T]) -> Matcher[T]:
    """Wraps argument in a matcher, if necessary.

    :returns: the argument as-is if it is already a matcher, otherwise wrapped
        in an :py:func:`~hamcrest.core.core.isequal.equal_to` matcher.

    """
    if isinstance(x, Matcher):
        return x
    else:
        return equal_to(x)