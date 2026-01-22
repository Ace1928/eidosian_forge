from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
def _undo(self):
    """
        _undo the changes of the most recent decision level.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> level = l._current_level
        >>> level.decision, level.var_settings, level.flipped
        (-3, {-3, -2}, False)
        >>> l._undo()
        >>> level = l._current_level
        >>> level.decision, level.var_settings, level.flipped
        (0, {1}, False)

        """
    for lit in self._current_level.var_settings:
        self.var_settings.remove(lit)
        self.heur_lit_unset(lit)
        self.variable_set[abs(lit)] = False
    self.levels.pop()