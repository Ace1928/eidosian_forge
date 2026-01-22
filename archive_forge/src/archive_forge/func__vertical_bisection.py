from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def _vertical_bisection(N, a, b, I, Q, F1, F2, f1, f2, F):
    """Vertical bisection step in Collins-Krandick root isolation algorithm. """
    (u, v), (s, t) = (a, b)
    I_L1, I_L2, I_L3, I_L4 = I
    Q_L1, Q_L2, Q_L3, Q_L4 = Q
    f1L1F, f1L2F, f1L3F, f1L4F = F1
    f2L1F, f2L2F, f2L3F, f2L4F = F2
    x = (u + s) / 2
    f1V = dmp_eval_in(f1, x, 0, 1, F)
    f2V = dmp_eval_in(f2, x, 0, 1, F)
    I_V = dup_isolate_real_roots_list([f1V, f2V], F, inf=v, sup=t, fast=True, strict=True, basis=True)
    I_L1_L, I_L1_R = ([], [])
    I_L2_L, I_L2_R = (I_V, I_L2)
    I_L3_L, I_L3_R = ([], [])
    I_L4_L, I_L4_R = (I_L4, _reverse_intervals(I_V))
    for I in I_L1:
        (a, b), indices, h = I
        if a == b:
            if a == x:
                I_L1_L.append(I)
                I_L1_R.append(I)
            elif a < x:
                I_L1_L.append(I)
            else:
                I_L1_R.append(I)
        elif b <= x:
            I_L1_L.append(I)
        elif a >= x:
            I_L1_R.append(I)
        else:
            a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=x, fast=True)
            if b <= x:
                I_L1_L.append(((a, b), indices, h))
            if a >= x:
                I_L1_R.append(((a, b), indices, h))
    for I in I_L3:
        (b, a), indices, h = I
        if a == b:
            if a == x:
                I_L3_L.append(I)
                I_L3_R.append(I)
            elif a < x:
                I_L3_L.append(I)
            else:
                I_L3_R.append(I)
        elif b <= x:
            I_L3_L.append(I)
        elif a >= x:
            I_L3_R.append(I)
        else:
            a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=x, fast=True)
            if b <= x:
                I_L3_L.append(((b, a), indices, h))
            if a >= x:
                I_L3_R.append(((b, a), indices, h))
    Q_L1_L = _intervals_to_quadrants(I_L1_L, f1L1F, f2L1F, u, x, F)
    Q_L2_L = _intervals_to_quadrants(I_L2_L, f1V, f2V, v, t, F)
    Q_L3_L = _intervals_to_quadrants(I_L3_L, f1L3F, f2L3F, x, u, F)
    Q_L4_L = Q_L4
    Q_L1_R = _intervals_to_quadrants(I_L1_R, f1L1F, f2L1F, x, s, F)
    Q_L2_R = Q_L2
    Q_L3_R = _intervals_to_quadrants(I_L3_R, f1L3F, f2L3F, s, x, F)
    Q_L4_R = _intervals_to_quadrants(I_L4_R, f1V, f2V, t, v, F)
    T_L = _traverse_quadrants(Q_L1_L, Q_L2_L, Q_L3_L, Q_L4_L, exclude=True)
    T_R = _traverse_quadrants(Q_L1_R, Q_L2_R, Q_L3_R, Q_L4_R, exclude=True)
    N_L = _winding_number(T_L, F)
    N_R = _winding_number(T_R, F)
    I_L = (I_L1_L, I_L2_L, I_L3_L, I_L4_L)
    Q_L = (Q_L1_L, Q_L2_L, Q_L3_L, Q_L4_L)
    I_R = (I_L1_R, I_L2_R, I_L3_R, I_L4_R)
    Q_R = (Q_L1_R, Q_L2_R, Q_L3_R, Q_L4_R)
    F1_L = (f1L1F, f1V, f1L3F, f1L4F)
    F2_L = (f2L1F, f2V, f2L3F, f2L4F)
    F1_R = (f1L1F, f1L2F, f1L3F, f1V)
    F2_R = (f2L1F, f2L2F, f2L3F, f2V)
    a, b = ((u, v), (x, t))
    c, d = ((x, v), (s, t))
    D_L = (N_L, a, b, I_L, Q_L, F1_L, F2_L)
    D_R = (N_R, c, d, I_R, Q_R, F1_R, F2_R)
    return (D_L, D_R)