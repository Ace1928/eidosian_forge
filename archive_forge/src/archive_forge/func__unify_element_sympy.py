from collections import defaultdict
from operator import index as index_
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer, Rational
from sympy.core.sympify import _sympify, SympifyError
from sympy.core.singleton import S
from sympy.polys.domains import ZZ, QQ, EXRAW
from sympy.polys.matrices import DomainMatrix
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent
from .common import classof
from .matrices import MatrixBase, MatrixKind, ShapeError
@classmethod
def _unify_element_sympy(cls, rep, element):
    domain = rep.domain
    element = _sympify(element)
    if domain != EXRAW:
        if element.is_Integer:
            new_domain = domain
        elif element.is_Rational:
            new_domain = QQ
        else:
            new_domain = EXRAW
        if new_domain != domain:
            rep = rep.convert_to(new_domain)
            domain = new_domain
        if domain != EXRAW:
            element = new_domain.from_sympy(element)
    if domain == EXRAW and (not isinstance(element, Expr)):
        sympy_deprecation_warning('\n                non-Expr objects in a Matrix is deprecated. Matrix represents\n                a mathematical matrix. To represent a container of non-numeric\n                entities, Use a list of lists, TableForm, NumPy array, or some\n                other data structure instead.\n                ', deprecated_since_version='1.9', active_deprecations_target='deprecated-non-expr-in-matrix', stacklevel=4)
    return (rep, element)