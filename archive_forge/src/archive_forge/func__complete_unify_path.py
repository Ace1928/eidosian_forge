import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def _complete_unify_path(first, second, bindings, used, skipped, debug):
    if used[0] or used[1]:
        newclause = Clause(skipped[0] + skipped[1] + first + second)
        debug.line('  -> New Clause: %s' % newclause)
        return [newclause.substitute_bindings(bindings)]
    else:
        debug.line('  -> End')
        return []