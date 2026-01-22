from sympy.core import Function, S, Mul, Pow, Add
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.function import expand_func
from sympy.core.symbol import Dummy
from sympy.functions import gamma, sqrt, sin
from sympy.polys import factor, cancel
from sympy.utilities.iterables import sift, uniq
def _mult_thm(gammas, numer, denom):
    rats = {}
    for g in gammas:
        c, resid = g.as_coeff_Add()
        rats.setdefault(resid, []).append(c)
    keys = sorted(rats, key=default_sort_key)
    for resid in keys:
        coeffs = sorted(rats[resid])
        new = []
        while True:
            run = _run(coeffs)
            if run is None:
                break
            n, ui, other = run
            for u in other:
                con = resid + u - 1
                for k in range(int(u - ui)):
                    numer.append(con - k)
            con = n * (resid + ui)
            numer.append((2 * S.Pi) ** (S(n - 1) / 2) * n ** (S.Half - con))
            new.append(con)
        rats[resid] = [resid + c for c in coeffs] + new
    g = []
    for resid in keys:
        g += rats[resid]
    gammas[:] = g