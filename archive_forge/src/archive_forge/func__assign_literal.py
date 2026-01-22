from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
def _assign_literal(self, lit):
    """Make a literal assignment.

        The literal assignment must be recorded as part of the current
        decision level. Additionally, if the literal is marked as a
        sentinel of any clause, then a new sentinel must be chosen. If
        this is not possible, then unit propagation is triggered and
        another literal is added to the queue to be set in the future.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> l.var_settings
        {-3, -2, 1}

        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> l._assign_literal(-1)
        >>> try:
        ...     next(l._find_model())
        ... except StopIteration:
        ...     pass
        >>> l.var_settings
        {-1}

        """
    self.var_settings.add(lit)
    self._current_level.var_settings.add(lit)
    self.variable_set[abs(lit)] = True
    self.heur_lit_assigned(lit)
    sentinel_list = list(self.sentinels[-lit])
    for cls in sentinel_list:
        if not self._clause_sat(cls):
            other_sentinel = None
            for newlit in self.clauses[cls]:
                if newlit != -lit:
                    if self._is_sentinel(newlit, cls):
                        other_sentinel = newlit
                    elif not self.variable_set[abs(newlit)]:
                        self.sentinels[-lit].remove(cls)
                        self.sentinels[newlit].add(cls)
                        other_sentinel = None
                        break
            if other_sentinel:
                self._unit_prop_queue.append(other_sentinel)