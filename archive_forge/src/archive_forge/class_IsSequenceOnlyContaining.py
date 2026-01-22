from typing import Sequence, TypeVar, Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.anyof import any_of
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
class IsSequenceOnlyContaining(BaseMatcher[Sequence[T]]):

    def __init__(self, matcher: Matcher[T]) -> None:
        self.matcher = matcher

    def _matches(self, item: Sequence[T]) -> bool:
        try:
            sequence = list(item)
            if len(sequence) == 0:
                return False
            for element in sequence:
                if not self.matcher.matches(element):
                    return False
            return True
        except TypeError:
            return False

    def describe_to(self, description: Description) -> None:
        description.append_text('a sequence containing items matching ').append_description_of(self.matcher)