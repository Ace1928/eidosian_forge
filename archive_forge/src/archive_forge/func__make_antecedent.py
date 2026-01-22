from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def _make_antecedent(self, predicate, signature):
    """
        Return an application expression with 'predicate' as the predicate
        and 'signature' as the list of arguments.
        """
    antecedent = predicate
    for v in signature:
        antecedent = antecedent(VariableExpression(v))
    return antecedent