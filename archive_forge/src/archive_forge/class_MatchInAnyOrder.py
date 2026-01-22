from typing import MutableSequence, Optional, Sequence, TypeVar, Union, cast
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
class MatchInAnyOrder(object):

    def __init__(self, matchers: Sequence[Matcher[T]], mismatch_description: Optional[Description]) -> None:
        self.matchers = cast(MutableSequence[Matcher[T]], matchers[:])
        self.mismatch_description = mismatch_description

    def matches(self, item: T) -> bool:
        return self.isnotsurplus(item) and self.ismatched(item)

    def isfinished(self, item: Sequence[T]) -> bool:
        if not self.matchers:
            return True
        if self.mismatch_description:
            self.mismatch_description.append_text('no item matches: ').append_list('', ', ', '', self.matchers).append_text(' in ').append_list('[', ', ', ']', item)
        return False

    def isnotsurplus(self, item: T) -> bool:
        if not self.matchers:
            if self.mismatch_description:
                self.mismatch_description.append_text('not matched: ').append_description_of(item)
            return False
        return True

    def ismatched(self, item: T) -> bool:
        for index, matcher in enumerate(self.matchers):
            if matcher.matches(item):
                del self.matchers[index]
                return True
        if self.mismatch_description:
            self.mismatch_description.append_text('not matched: ').append_description_of(item)
        return False