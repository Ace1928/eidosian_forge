from __future__ import annotations
from sympy.vector.basisdependent import (BasisDependent, BasisDependentAdd,
from sympy.core import S, Pow
from sympy.core.expr import AtomicExpr
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
import sympy.vector
class DyadicZero(BasisDependentZero, Dyadic):
    """
    Class to denote a zero dyadic
    """
    _op_priority = 13.1
    _pretty_form = '(0|0)'
    _latex_form = '(\\mathbf{\\hat{0}}|\\mathbf{\\hat{0}})'

    def __new__(cls):
        obj = BasisDependentZero.__new__(cls)
        return obj