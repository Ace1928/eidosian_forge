import re
from typing import Any, Optional, Tuple
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
class DescribedAs(BaseMatcher[Any]):

    def __init__(self, description_template: str, matcher: Matcher[Any], *values: Tuple[Any, ...]) -> None:
        self.template = description_template
        self.matcher = matcher
        self.values = values

    def matches(self, item: Any, mismatch_description: Optional[Description]=None) -> bool:
        return self.matcher.matches(item, mismatch_description)

    def describe_mismatch(self, item: Any, mismatch_description: Description) -> None:
        self.matcher.describe_mismatch(item, mismatch_description)

    def describe_to(self, description: Description) -> None:
        text_start = 0
        for match in re.finditer(ARG_PATTERN, self.template):
            description.append_text(self.template[text_start:match.start()])
            arg_index = int(match.group()[1:])
            description.append_description_of(self.values[arg_index])
            text_start = match.end()
        if text_start < len(self.template):
            description.append_text(self.template[text_start:])