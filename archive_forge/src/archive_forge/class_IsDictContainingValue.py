from typing import Any, Mapping, TypeVar, Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
class IsDictContainingValue(BaseMatcher[Mapping[Any, V]]):

    def __init__(self, value_matcher: Matcher[V]) -> None:
        self.value_matcher = value_matcher

    def _matches(self, item: Mapping[Any, V]) -> bool:
        if hasmethod(item, 'values'):
            for value in item.values():
                if self.value_matcher.matches(value):
                    return True
        return False

    def describe_to(self, description: Description) -> None:
        description.append_text('a dictionary containing value ').append_description_of(self.value_matcher)