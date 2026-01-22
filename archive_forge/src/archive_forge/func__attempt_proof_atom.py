from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _attempt_proof_atom(self, current, context, agenda, accessible_vars, atoms, debug):
    if (current, True) in atoms:
        debug.line('CLOSED', 1)
        return True
    if context:
        if isinstance(context.term, NegatedExpression):
            current = current.negate()
        agenda.put(context(current).simplify())
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
    else:
        agenda.mark_alls_fresh()
        return self._attempt_proof(agenda, accessible_vars | set(current.args), atoms | {(current, False)}, debug + 1)