from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
class Prover:
    """ai - prover of logic rules

       given a set of initial rules, Prover tries to prove all possible rules
       which follow from given premises.

       As a result proved_rules are always either in one of two forms: alpha or
       beta:

       Alpha rules
       -----------

       This are rules of the form::

         a -> b & c & d & ...


       Beta rules
       ----------

       This are rules of the form::

         &(a,b,...) -> c & d & ...


       i.e. beta rules are join conditions that say that something follows when
       *several* facts are true at the same time.
    """

    def __init__(self):
        self.proved_rules = []
        self._rules_seen = set()

    def split_alpha_beta(self):
        """split proved rules into alpha and beta chains"""
        rules_alpha = []
        rules_beta = []
        for a, b in self.proved_rules:
            if isinstance(a, And):
                rules_beta.append((a, b))
            else:
                rules_alpha.append((a, b))
        return (rules_alpha, rules_beta)

    @property
    def rules_alpha(self):
        return self.split_alpha_beta()[0]

    @property
    def rules_beta(self):
        return self.split_alpha_beta()[1]

    def process_rule(self, a, b):
        """process a -> b rule"""
        if not a or isinstance(b, bool):
            return
        if isinstance(a, bool):
            return
        if (a, b) in self._rules_seen:
            return
        else:
            self._rules_seen.add((a, b))
        try:
            self._process_rule(a, b)
        except TautologyDetected:
            pass

    def _process_rule(self, a, b):
        if isinstance(b, And):
            sorted_bargs = sorted(b.args, key=str)
            for barg in sorted_bargs:
                self.process_rule(a, barg)
        elif isinstance(b, Or):
            sorted_bargs = sorted(b.args, key=str)
            if not isinstance(a, Logic):
                if a in sorted_bargs:
                    raise TautologyDetected(a, b, 'a -> a|c|...')
            self.process_rule(And(*[Not(barg) for barg in b.args]), Not(a))
            for bidx in range(len(sorted_bargs)):
                barg = sorted_bargs[bidx]
                brest = sorted_bargs[:bidx] + sorted_bargs[bidx + 1:]
                self.process_rule(And(a, Not(barg)), Or(*brest))
        elif isinstance(a, And):
            sorted_aargs = sorted(a.args, key=str)
            if b in sorted_aargs:
                raise TautologyDetected(a, b, 'a & b -> a')
            self.proved_rules.append((a, b))
        elif isinstance(a, Or):
            sorted_aargs = sorted(a.args, key=str)
            if b in sorted_aargs:
                raise TautologyDetected(a, b, 'a | b -> a')
            for aarg in sorted_aargs:
                self.process_rule(aarg, b)
        else:
            self.proved_rules.append((a, b))
            self.proved_rules.append((Not(b), Not(a)))