from typing import Any, List, Sequence, Tuple, TypeVar
from hamcrest import (
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import AllOf
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from typing_extensions import Protocol
from twisted.python.failure import Failure
class Semigroup(Protocol[T]):
    """
    A type with an associative binary operator.

    Common examples of a semigroup are integers with addition and strings with
    concatenation.
    """

    def __add__(self, other: T) -> T:
        """
        This must be associative: a + (b + c) == (a + b) + c
        """