from sympy.core.sympify import sympify
from sympy.core import (S, Pow, Dummy, pi, Expr, Wild, Mul, Equality,
from sympy.core.containers import Tuple
from sympy.core.function import (Lambda, expand_complex, AppliedUndef,
from sympy.core.mod import Mod
from sympy.core.numbers import igcd, I, Number, Rational, oo, ilcm
from sympy.core.power import integer_log
from sympy.core.relational import Eq, Ne, Relational
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, _uniquely_named_symbol
from sympy.core.sympify import _sympify
from sympy.polys.matrices.linsolve import _linear_eq_to_dict
from sympy.polys.polyroots import UnsolvableFactorError
from sympy.simplify.simplify import simplify, fraction, trigsimp, nsimplify
from sympy.simplify import powdenest, logcombine
from sympy.functions import (log, tan, cot, sin, cos, sec, csc, exp,
from sympy.functions.elementary.complexes import Abs, arg, re, im
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.miscellaneous import real_root
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import And, BooleanTrue
from sympy.sets import (FiniteSet, imageset, Interval, Intersection,
from sympy.sets.sets import Set, ProductSet
from sympy.matrices import zeros, Matrix, MatrixBase
from sympy.ntheory import totient
from sympy.ntheory.factor_ import divisors
from sympy.ntheory.residue_ntheory import discrete_log, nthroot_mod
from sympy.polys import (roots, Poly, degree, together, PolynomialError,
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import invert, groebner, poly
from sympy.polys.solvers import (sympy_eqs_to_ring, solve_lin_sys,
from sympy.polys.matrices.linsolve import _linsolve
from sympy.solvers.solvers import (checksol, denoms, unrad,
from sympy.solvers.polysys import solve_poly_system
from sympy.utilities import filldedent
from sympy.utilities.iterables import (numbered_symbols, has_dups,
from sympy.calculus.util import periodicity, continuous_domain, function_range
from types import GeneratorType
def add_intersection_complement(result, intersection_dict, complement_dict):
    final_result = []
    for res in result:
        res_copy = res
        for key_res, value_res in res.items():
            intersect_set, complement_set = (None, None)
            for key_sym, value_sym in intersection_dict.items():
                if key_sym == key_res:
                    intersect_set = value_sym
            for key_sym, value_sym in complement_dict.items():
                if key_sym == key_res:
                    complement_set = value_sym
            if intersect_set or complement_set:
                new_value = FiniteSet(value_res)
                if intersect_set and intersect_set != S.Complexes:
                    new_value = Intersection(new_value, intersect_set)
                if complement_set:
                    new_value = Complement(new_value, complement_set)
                if new_value is S.EmptySet:
                    res_copy = None
                    break
                elif new_value.is_FiniteSet and len(new_value) == 1:
                    res_copy[key_res] = set(new_value).pop()
                else:
                    res_copy[key_res] = new_value
        if res_copy is not None:
            final_result.append(res_copy)
    return final_result