import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def _prove(self, goal=None, assumptions=None, verbose=False):
    """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in the proof
        :type assumptions: list(sem.Expression)
        """
    if not assumptions:
        assumptions = []
    result = None
    try:
        clauses = []
        if goal:
            clauses.extend(clausify(-goal))
        for a in assumptions:
            clauses.extend(clausify(a))
        result, clauses = self._attempt_proof(clauses)
        if verbose:
            print(ResolutionProverCommand._decorate_clauses(clauses))
    except RuntimeError as e:
        if self._assume_false and str(e).startswith('maximum recursion depth exceeded'):
            result = False
            clauses = []
        elif verbose:
            print(e)
        else:
            raise e
    return (result, clauses)