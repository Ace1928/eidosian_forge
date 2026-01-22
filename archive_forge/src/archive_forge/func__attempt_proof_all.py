from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _attempt_proof_all(self, current, context, agenda, accessible_vars, atoms, debug):
    try:
        current._used_vars
    except AttributeError:
        current._used_vars = set()
    if accessible_vars:
        bv_available = accessible_vars - current._used_vars
        if bv_available:
            variable_to_use = list(bv_available)[0]
            debug.line("--> Using '%s'" % variable_to_use, 2)
            current._used_vars |= {variable_to_use}
            agenda.put(current.term.replace(current.variable, variable_to_use), context)
            agenda[Categories.ALL].add((current, context))
            return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
        else:
            debug.line('--> Variables Exhausted', 2)
            current._exhausted = True
            agenda[Categories.ALL].add((current, context))
            return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
    else:
        new_unique_variable = VariableExpression(unique_variable())
        debug.line("--> Using '%s'" % new_unique_variable, 2)
        current._used_vars |= {new_unique_variable}
        agenda.put(current.term.replace(current.variable, new_unique_variable), context)
        agenda[Categories.ALL].add((current, context))
        agenda.mark_alls_fresh()
        return self._attempt_proof(agenda, accessible_vars | {new_unique_variable}, atoms, debug + 1)