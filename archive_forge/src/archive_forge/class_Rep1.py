from __future__ import absolute_import
import types
from . import Errors
class Rep1(RE):
    """Rep1(re) is an RE which matches one or more repetitions of |re|."""

    def __init__(self, re):
        self.check_re(1, re)
        self.re = re
        self.nullable = re.nullable
        self.match_nl = re.match_nl

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        s1 = m.new_state()
        s2 = m.new_state()
        initial_state.link_to(s1)
        self.re.build_machine(m, s1, s2, match_bol or self.re.match_nl, nocase)
        s2.link_to(s1)
        s2.link_to(final_state)

    def calc_str(self):
        return 'Rep1(%s)' % self.re