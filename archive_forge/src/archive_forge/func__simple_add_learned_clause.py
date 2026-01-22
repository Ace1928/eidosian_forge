from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
def _simple_add_learned_clause(self, cls):
    """Add a new clause to the theory.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())

        >>> l.num_learned_clauses
        0
        >>> l.clauses
        [[2, -3], [1], [3, -3], [2, -2], [3, -2]]
        >>> l.sentinels
        {-3: {0, 2}, -2: {3, 4}, 2: {0, 3}, 3: {2, 4}}

        >>> l._simple_add_learned_clause([3])

        >>> l.clauses
        [[2, -3], [1], [3, -3], [2, -2], [3, -2], [3]]
        >>> l.sentinels
        {-3: {0, 2}, -2: {3, 4}, 2: {0, 3}, 3: {2, 4, 5}}

        """
    cls_num = len(self.clauses)
    self.clauses.append(cls)
    for lit in cls:
        self.occurrence_count[lit] += 1
    self.sentinels[cls[0]].add(cls_num)
    self.sentinels[cls[-1]].add(cls_num)
    self.heur_clause_added(cls)