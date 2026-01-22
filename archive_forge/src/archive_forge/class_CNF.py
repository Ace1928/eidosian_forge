from itertools import combinations, product, zip_longest
from sympy.assumptions.assume import AppliedPredicate, Predicate
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.logic.boolalg import Or, And, Not, Xnor
from sympy.logic.boolalg import (Equivalent, ITE, Implies, Nand, Nor, Xor)
class CNF:
    """
    Class to represent CNF of a Boolean expression.
    Consists of set of clauses, which themselves are stored as
    frozenset of Literal objects.

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.abc import x
    >>> cnf = CNF.from_prop(Q.real(x) & ~Q.zero(x))
    >>> cnf.clauses
    {frozenset({Literal(Q.zero(x), True)}),
    frozenset({Literal(Q.negative(x), False),
    Literal(Q.positive(x), False), Literal(Q.zero(x), False)})}
    """

    def __init__(self, clauses=None):
        if not clauses:
            clauses = set()
        self.clauses = clauses

    def add(self, prop):
        clauses = CNF.to_CNF(prop).clauses
        self.add_clauses(clauses)

    def __str__(self):
        s = ' & '.join(['(' + ' | '.join([str(lit) for lit in clause]) + ')' for clause in self.clauses])
        return s

    def extend(self, props):
        for p in props:
            self.add(p)
        return self

    def copy(self):
        return CNF(set(self.clauses))

    def add_clauses(self, clauses):
        self.clauses |= clauses

    @classmethod
    def from_prop(cls, prop):
        res = cls()
        res.add(prop)
        return res

    def __iand__(self, other):
        self.add_clauses(other.clauses)
        return self

    def all_predicates(self):
        predicates = set()
        for c in self.clauses:
            predicates |= {arg.lit for arg in c}
        return predicates

    def _or(self, cnf):
        clauses = set()
        for a, b in product(self.clauses, cnf.clauses):
            tmp = set(a)
            for t in b:
                tmp.add(t)
            clauses.add(frozenset(tmp))
        return CNF(clauses)

    def _and(self, cnf):
        clauses = self.clauses.union(cnf.clauses)
        return CNF(clauses)

    def _not(self):
        clss = list(self.clauses)
        ll = set()
        for x in clss[-1]:
            ll.add(frozenset((~x,)))
        ll = CNF(ll)
        for rest in clss[:-1]:
            p = set()
            for x in rest:
                p.add(frozenset((~x,)))
            ll = ll._or(CNF(p))
        return ll

    def rcall(self, expr):
        clause_list = []
        for clause in self.clauses:
            lits = [arg.rcall(expr) for arg in clause]
            clause_list.append(OR(*lits))
        expr = AND(*clause_list)
        return distribute_AND_over_OR(expr)

    @classmethod
    def all_or(cls, *cnfs):
        b = cnfs[0].copy()
        for rest in cnfs[1:]:
            b = b._or(rest)
        return b

    @classmethod
    def all_and(cls, *cnfs):
        b = cnfs[0].copy()
        for rest in cnfs[1:]:
            b = b._and(rest)
        return b

    @classmethod
    def to_CNF(cls, expr):
        from sympy.assumptions.facts import get_composite_predicates
        expr = to_NNF(expr, get_composite_predicates())
        expr = distribute_AND_over_OR(expr)
        return expr

    @classmethod
    def CNF_to_cnf(cls, cnf):
        """
        Converts CNF object to SymPy's boolean expression
        retaining the form of expression.
        """

        def remove_literal(arg):
            return Not(arg.lit) if arg.is_Not else arg.lit
        return And(*(Or(*(remove_literal(arg) for arg in clause)) for clause in cnf.clauses))