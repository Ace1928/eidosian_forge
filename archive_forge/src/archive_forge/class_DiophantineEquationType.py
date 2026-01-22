from sympy.core.add import Add
from sympy.core.assumptions import check_assumptions
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.numbers import igcdex, ilcm, igcd
from sympy.core.power import integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import (
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solveset import solveset_real
from sympy.utilities import numbered_symbols
from sympy.utilities.misc import as_int, filldedent
from sympy.utilities.iterables import (is_sequence, subsets, permute_signs,
class DiophantineEquationType:
    """
    Internal representation of a particular diophantine equation type.

    Parameters
    ==========

    equation :
        The diophantine equation that is being solved.
    free_symbols : list (optional)
        The symbols being solved for.

    Attributes
    ==========

    total_degree :
        The maximum of the degrees of all terms in the equation
    homogeneous :
        Does the equation contain a term of degree 0
    homogeneous_order :
        Does the equation contain any coefficient that is in the symbols being solved for
    dimension :
        The number of symbols being solved for
    """
    name = None

    def __init__(self, equation, free_symbols=None):
        self.equation = _sympify(equation).expand(force=True)
        if free_symbols is not None:
            self.free_symbols = free_symbols
        else:
            self.free_symbols = list(self.equation.free_symbols)
            self.free_symbols.sort(key=default_sort_key)
        if not self.free_symbols:
            raise ValueError('equation should have 1 or more free symbols')
        self.coeff = self.equation.as_coefficients_dict()
        if not all((_is_int(c) for c in self.coeff.values())):
            raise TypeError('Coefficients should be Integers')
        self.total_degree = Poly(self.equation).total_degree()
        self.homogeneous = 1 not in self.coeff
        self.homogeneous_order = not set(self.coeff) & set(self.free_symbols)
        self.dimension = len(self.free_symbols)
        self._parameters = None

    def matches(self):
        """
        Determine whether the given equation can be matched to the particular equation type.
        """
        return False

    @property
    def n_parameters(self):
        return self.dimension

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = symbols('t_:%i' % (self.n_parameters,), integer=True)
        return self._parameters

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        raise NotImplementedError('No solver has been written for %s.' % self.name)

    def pre_solve(self, parameters=None):
        if not self.matches():
            raise ValueError('This equation does not match the %s equation type.' % self.name)
        if parameters is not None:
            if len(parameters) != self.n_parameters:
                raise ValueError('Expected %s parameter(s) but got %s' % (self.n_parameters, len(parameters)))
        self._parameters = parameters