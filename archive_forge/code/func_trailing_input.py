from __future__ import unicode_literals
import re
from six.moves import range
from .regex_parser import Any, Sequence, Regex, Variable, Repeat, Lookahead
from .regex_parser import parse_regex, tokenize_regex
def trailing_input(self):
    """
        Get the `MatchVariable` instance, representing trailing input, if there is any.
        "Trailing input" is input at the end that does not match the grammar anymore, but
        when this is removed from the end of the input, the input would be a valid string.
        """
    slices = []
    for r, re_match in self._re_matches:
        for group_name, group_index in r.groupindex.items():
            if group_name == _INVALID_TRAILING_INPUT:
                slices.append(re_match.regs[group_index])
    if slices:
        slice = [max((i[0] for i in slices)), max((i[1] for i in slices))]
        value = self.string[slice[0]:slice[1]]
        return MatchVariable('<trailing_input>', value, slice)