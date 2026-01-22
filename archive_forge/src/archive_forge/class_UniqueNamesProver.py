from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
class UniqueNamesProver(ProverCommandDecorator):
    """
    This is a prover decorator that adds unique names assumptions before
    proving.
    """

    def assumptions(self):
        """
        - Domain = union([e.free()|e.constants() for e in all_expressions])
        - if "d1 = d2" cannot be proven from the premises, then add "d1 != d2"
        """
        assumptions = self._command.assumptions()
        domain = list(get_domain(self._command.goal(), assumptions))
        eq_sets = SetHolder()
        for a in assumptions:
            if isinstance(a, EqualityExpression):
                av = a.first.variable
                bv = a.second.variable
                eq_sets[av].add(bv)
        new_assumptions = []
        for i, a in enumerate(domain):
            for b in domain[i + 1:]:
                if b not in eq_sets[a]:
                    newEqEx = EqualityExpression(VariableExpression(a), VariableExpression(b))
                    if Prover9().prove(newEqEx, assumptions):
                        eq_sets[a].add(b)
                    else:
                        new_assumptions.append(-newEqEx)
        return assumptions + new_assumptions