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
def _solve_using_known_values(result, solver):
    """Solves the system using already known solution
        (result contains the dict <symbol: value>).
        solver is :func:`~.solveset_complex` or :func:`~.solveset_real`.
        """
    soln_imageset = {}
    total_solvest_call = 0
    total_conditionst = 0
    for index, eq in enumerate(eqs_in_better_order):
        newresult = []
        original_imageset = {}
        imgset_yes = False
        result = _new_order_result(result, eq)
        for res in result:
            got_symbol = set()
            for key_res, value_res in res.items():
                if isinstance(value_res, ImageSet):
                    res[key_res] = value_res.lamda.expr
                    original_imageset[key_res] = value_res
                    dummy_n = value_res.lamda.expr.atoms(Dummy).pop()
                    base, = value_res.base_sets
                    imgset_yes = (dummy_n, base)
            eq2 = eq.subs(res).expand()
            unsolved_syms = _unsolved_syms(eq2, sort=True)
            if not unsolved_syms:
                if res:
                    newresult, delete_res = _append_new_soln(res, None, None, imgset_yes, soln_imageset, original_imageset, newresult, eq2)
                    if delete_res:
                        result.remove(res)
                continue
            depen1, depen2 = eq2.rewrite(Add).as_independent(*unsolved_syms)
            if (depen1.has(Abs) or depen2.has(Abs)) and solver == solveset_complex:
                continue
            soln_imageset = {}
            for sym in unsolved_syms:
                not_solvable = False
                try:
                    soln = solver(eq2, sym)
                    total_solvest_call += 1
                    soln_new = S.EmptySet
                    if isinstance(soln, Complement):
                        complements[sym] = soln.args[1]
                        soln = soln.args[0]
                    if isinstance(soln, Intersection):
                        if soln.args[0] != Interval(-oo, oo):
                            intersections[sym] = soln.args[0]
                        soln_new += soln.args[1]
                    soln = soln_new if soln_new else soln
                    if index > 0 and solver == solveset_real:
                        if not isinstance(soln, (ImageSet, ConditionSet)):
                            soln += solveset_complex(eq2, sym)
                except (NotImplementedError, ValueError):
                    continue
                if isinstance(soln, ConditionSet):
                    if soln.base_set in (S.Reals, S.Complexes):
                        soln = S.EmptySet
                        not_solvable = True
                        total_conditionst += 1
                    else:
                        soln = soln.base_set
                if soln is not S.EmptySet:
                    soln, soln_imageset = _extract_main_soln(sym, soln, soln_imageset)
                for sol in soln:
                    sol, soln_imageset = _extract_main_soln(sym, sol, soln_imageset)
                    sol = set(sol).pop()
                    free = sol.free_symbols
                    if got_symbol and any((ss in free for ss in got_symbol)):
                        continue
                    rnew = res.copy()
                    for k, v in res.items():
                        if isinstance(v, Expr) and isinstance(sol, Expr):
                            rnew[k] = v.subs(sym, sol)
                    if sol in soln_imageset.keys():
                        imgst = soln_imageset[sol]
                        rnew[sym] = imgst.lamda(*[0 for i in range(0, len(imgst.lamda.variables))])
                    else:
                        rnew[sym] = sol
                    newresult, delete_res = _append_new_soln(rnew, sym, sol, imgset_yes, soln_imageset, original_imageset, newresult)
                    if delete_res:
                        result.remove(res)
                if not not_solvable:
                    got_symbol.add(sym)
        if newresult:
            result = newresult
    return (result, total_solvest_call, total_conditionst)