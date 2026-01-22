from typing import Any
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
class IsEqual(BaseMatcher[Any]):

    def __init__(self, equals: Any) -> None:
        self.object = equals

    def _matches(self, item: Any) -> bool:
        return item == self.object

    def describe_to(self, description: Description) -> None:
        nested_matcher = isinstance(self.object, Matcher)
        if nested_matcher:
            description.append_text('<')
        description.append_description_of(self.object)
        if nested_matcher:
            description.append_text('>')