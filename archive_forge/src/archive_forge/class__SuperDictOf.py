from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
class _SuperDictOf(Matcher):
    """Matches if all of the keys in the given dict are in the matched dict.
    """

    def __init__(self, sub_dict, format_value=repr):
        super().__init__()
        self.sub_dict = sub_dict
        self.format_value = format_value

    def match(self, super_dict):
        return _SubDictOf(super_dict, self.format_value).match(self.sub_dict)