from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
class HasString(BaseMatcher[object]):

    def __init__(self, str_matcher: Matcher[str]) -> None:
        self.str_matcher = str_matcher

    def _matches(self, item: object) -> bool:
        return self.str_matcher.matches(str(item))

    def describe_to(self, description: Description) -> None:
        description.append_text('an object with str ').append_description_of(self.str_matcher)