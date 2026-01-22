from __future__ import absolute_import
import types
from . import Errors
class RawCodeRange(RE):
    """
    RawCodeRange(code1, code2) is a low-level RE which matches any character
    with a code |c| in the range |code1| <= |c| < |code2|, where the range
    does not include newline. For internal use only.
    """
    nullable = 0
    match_nl = 0
    range = None
    uppercase_range = None
    lowercase_range = None

    def __init__(self, code1, code2):
        self.range = (code1, code2)
        self.uppercase_range = uppercase_range(code1, code2)
        self.lowercase_range = lowercase_range(code1, code2)

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        if match_bol:
            initial_state = self.build_opt(m, initial_state, BOL)
        initial_state.add_transition(self.range, final_state)
        if nocase:
            if self.uppercase_range:
                initial_state.add_transition(self.uppercase_range, final_state)
            if self.lowercase_range:
                initial_state.add_transition(self.lowercase_range, final_state)

    def calc_str(self):
        return 'CodeRange(%d,%d)' % (self.code1, self.code2)