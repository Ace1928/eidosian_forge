from typing import Any, Optional
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
class IsAnything(BaseMatcher[Any]):

    def __init__(self, description: Optional[str]) -> None:
        self.description: str = description or 'ANYTHING'

    def _matches(self, item: Any) -> bool:
        return True

    def describe_to(self, description: Description) -> None:
        description.append_text(self.description)