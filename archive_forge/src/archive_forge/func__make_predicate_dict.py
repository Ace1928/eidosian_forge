from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def _make_predicate_dict(self, assumptions):
    """
        Create a dictionary of predicates from the assumptions.

        :param assumptions: a list of ``Expression``s
        :return: dict mapping ``AbstractVariableExpression`` to ``PredHolder``
        """
    predicates = defaultdict(PredHolder)
    for a in assumptions:
        self._map_predicates(a, predicates)
    return predicates