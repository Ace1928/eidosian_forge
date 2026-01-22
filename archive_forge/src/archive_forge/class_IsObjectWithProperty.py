from typing import Any, Mapping, TypeVar, Union, overload
from hamcrest import described_as
from hamcrest.core import anything
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import AllOf
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher as wrap_shortcut
from hamcrest.core.matcher import Matcher
from hamcrest.core.string_description import StringDescription
class IsObjectWithProperty(BaseMatcher[object]):

    def __init__(self, property_name: str, value_matcher: Matcher[V]) -> None:
        self.property_name = property_name
        self.value_matcher = value_matcher

    def _matches(self, item: object) -> bool:
        if item is None:
            return False
        if not hasattr(item, self.property_name):
            return False
        value = getattr(item, self.property_name)
        return self.value_matcher.matches(value)

    def describe_to(self, description: Description) -> None:
        description.append_text("an object with a property '").append_text(self.property_name).append_text("' matching ").append_description_of(self.value_matcher)

    def describe_mismatch(self, item: object, mismatch_description: Description) -> None:
        if item is None:
            mismatch_description.append_text('was None')
            return
        if not hasattr(item, self.property_name):
            mismatch_description.append_description_of(item).append_text(' did not have the ').append_description_of(self.property_name).append_text(' property')
            return
        mismatch_description.append_text('property ').append_description_of(self.property_name).append_text(' ')
        value = getattr(item, self.property_name)
        self.value_matcher.describe_mismatch(value, mismatch_description)

    def __str__(self):
        d = StringDescription()
        self.describe_to(d)
        return str(d)