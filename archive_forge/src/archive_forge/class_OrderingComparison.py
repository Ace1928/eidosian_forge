import operator
from typing import Any, Callable
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
class OrderingComparison(BaseMatcher[Any]):

    def __init__(self, value: Any, comparison_function: Callable[[Any, Any], bool], comparison_description: str) -> None:
        self.value = value
        self.comparison_function = comparison_function
        self.comparison_description = comparison_description

    def _matches(self, item: Any) -> bool:
        try:
            return self.comparison_function(item, self.value)
        except TypeError:
            return False

    def describe_to(self, description: Description) -> None:
        description.append_text('a value ').append_text(self.comparison_description).append_text(' ').append_description_of(self.value)