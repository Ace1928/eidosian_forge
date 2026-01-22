from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def _make_unique_signature(self, predHolder):
    """
        This method figures out how many arguments the predicate takes and
        returns a tuple containing that number of unique variables.
        """
    return tuple((unique_variable() for i in range(predHolder.signature_len)))