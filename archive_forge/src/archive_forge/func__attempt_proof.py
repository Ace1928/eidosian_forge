import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def _attempt_proof(self, clauses):
    tried = defaultdict(list)
    i = 0
    while i < len(clauses):
        if not clauses[i].is_tautology():
            if tried[i]:
                j = tried[i][-1] + 1
            else:
                j = i + 1
            while j < len(clauses):
                if i != j and j and (not clauses[j].is_tautology()):
                    tried[i].append(j)
                    newclauses = clauses[i].unify(clauses[j])
                    if newclauses:
                        for newclause in newclauses:
                            newclause._parents = (i + 1, j + 1)
                            clauses.append(newclause)
                            if not len(newclause):
                                return (True, clauses)
                        i = -1
                        break
                j += 1
        i += 1
    return (False, clauses)