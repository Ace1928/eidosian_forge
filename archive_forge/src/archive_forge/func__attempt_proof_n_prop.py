from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _attempt_proof_n_prop(self, current, context, agenda, accessible_vars, atoms, debug):
    if (current.term, False) in atoms:
        debug.line('CLOSED', 1)
        return True
    agenda.mark_alls_fresh()
    return self._attempt_proof(agenda, accessible_vars, atoms | {(current.term, True)}, debug + 1)