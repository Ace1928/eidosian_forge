from __future__ import absolute_import
import types
from . import Errors
class SpecialSymbol(RE):
    """
    SpecialSymbol(sym) is an RE which matches the special input
    symbol |sym|, which is one of BOL, EOL or EOF.
    """
    nullable = 0
    match_nl = 0
    sym = None

    def __init__(self, sym):
        self.sym = sym

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        if match_bol and self.sym == EOL:
            initial_state = self.build_opt(m, initial_state, BOL)
        initial_state.add_transition(self.sym, final_state)