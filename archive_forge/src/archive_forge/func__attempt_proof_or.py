from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _attempt_proof_or(self, current, context, agenda, accessible_vars, atoms, debug):
    new_agenda = agenda.clone()
    agenda.put(current.first, context)
    new_agenda.put(current.second, context)
    return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1) and self._attempt_proof(new_agenda, accessible_vars, atoms, debug + 1)