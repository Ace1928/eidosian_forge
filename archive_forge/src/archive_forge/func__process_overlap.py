from collections import deque
from sympy.combinatorics.rewritingsystem_fsm import StateMachine
def _process_overlap(w, r1, r2, check):
    s = w.eliminate_word(r1, self.rules[r1])
    s = self.reduce(s)
    t = w.eliminate_word(r2, self.rules[r2])
    t = self.reduce(t)
    if s != t:
        if check:
            return [0]
        try:
            new_keys = self.add_rule(t, s, check)
            return new_keys
        except RuntimeError:
            return False
    return