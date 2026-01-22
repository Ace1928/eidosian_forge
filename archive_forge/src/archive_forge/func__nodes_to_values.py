from __future__ import unicode_literals
import re
from six.moves import range
from .regex_parser import Any, Sequence, Regex, Variable, Repeat, Lookahead
from .regex_parser import parse_regex, tokenize_regex
def _nodes_to_values(self):
    """
        Returns list of list of (Node, string_value) tuples.
        """

    def is_none(slice):
        return slice[0] == -1 and slice[1] == -1

    def get(slice):
        return self.string[slice[0]:slice[1]]
    return [(varname, get(slice), slice) for varname, slice in self._nodes_to_regs() if not is_none(slice)]