from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
class IsEqualIgnoringCase(BaseMatcher[str]):

    def __init__(self, string: str) -> None:
        if not isinstance(string, str):
            raise TypeError('IsEqualIgnoringCase requires string')
        self.original_string = string
        self.lowered_string = string.lower()

    def _matches(self, item: str) -> bool:
        if not isinstance(item, str):
            return False
        return self.lowered_string == item.lower()

    def describe_to(self, description: Description) -> None:
        description.append_description_of(self.original_string).append_text(' ignoring case')