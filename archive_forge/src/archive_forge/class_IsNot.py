from typing import Type, TypeVar, Union, overload
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import is_matchable_type, wrap_matcher
from hamcrest.core.matcher import Matcher
from .isinstanceof import instance_of
class IsNot(BaseMatcher[T]):

    def __init__(self, matcher: Matcher[T]) -> None:
        self.matcher = matcher

    def _matches(self, item: T) -> bool:
        return not self.matcher.matches(item)

    def describe_to(self, description: Description) -> None:
        description.append_text('not ').append_description_of(self.matcher)

    def describe_mismatch(self, item: T, mismatch_description: Description) -> None:
        mismatch_description.append_text('but ')
        self.matcher.describe_match(item, mismatch_description)