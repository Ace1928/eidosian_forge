from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _attempt_proof_and(self, current, context, agenda, accessible_vars, atoms, debug):
    agenda.put(current.first, context)
    agenda.put(current.second, context)
    return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)