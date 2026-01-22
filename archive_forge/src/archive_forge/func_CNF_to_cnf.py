from itertools import combinations, product, zip_longest
from sympy.assumptions.assume import AppliedPredicate, Predicate
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.logic.boolalg import Or, And, Not, Xnor
from sympy.logic.boolalg import (Equivalent, ITE, Implies, Nand, Nor, Xor)
@classmethod
def CNF_to_cnf(cls, cnf):
    """
        Converts CNF object to SymPy's boolean expression
        retaining the form of expression.
        """

    def remove_literal(arg):
        return Not(arg.lit) if arg.is_Not else arg.lit
    return And(*(Or(*(remove_literal(arg) for arg in clause)) for clause in cnf.clauses))