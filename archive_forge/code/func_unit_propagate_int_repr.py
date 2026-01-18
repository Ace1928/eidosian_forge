from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Or, Not, conjuncts, disjuncts, to_cnf, \
from sympy.assumptions.cnf import CNF
from sympy.logic.inference import pl_true, literal_symbol
def unit_propagate_int_repr(clauses, s):
    """
    Same as unit_propagate, but arguments are expected to be in integer
    representation

    >>> from sympy.logic.algorithms.dpll import unit_propagate_int_repr
    >>> unit_propagate_int_repr([{1, 2}, {3, -2}, {2}], 2)
    [{3}]

    """
    negated = {-s}
    return [clause - negated for clause in clauses if s not in clause]