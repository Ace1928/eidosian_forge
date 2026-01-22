from typing import List, Tuple
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import (
from cvxpy.constraints.zero import Zero
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers.von_neumann_entr_canon import (
def OpRelEntrConeQuad_canon(con: OpRelEntrConeQuad, args) -> Tuple[Constraint, List[Constraint]]:
    k, m = (con.k, con.m)
    X, Y = (con.X, con.Y)
    assert X.is_real()
    assert Y.is_real()
    assert con.Z.is_real()
    Zs = {i: Variable(shape=X.shape, symmetric=True) for i in range(k + 1)}
    Ts = {i: Variable(shape=X.shape, symmetric=True) for i in range(m + 1)}
    constrs = [Zero(Zs[0] - Y)]
    if not X.is_symmetric():
        ut = upper_tri(X)
        lt = upper_tri(X.T)
        constrs.append(ut == lt)
    if not Y.is_symmetric():
        ut = upper_tri(Y)
        lt = upper_tri(Y.T)
        constrs.append(ut == lt)
    if not con.Z.is_symmetric():
        ut = upper_tri(con.Z)
        lt = upper_tri(con.Z.T)
        constrs.append(ut == lt)
    w, t = gauss_legendre(m)
    lead_con = Zero(cp.sum([w[i] * Ts[i] for i in range(m)]) + con.Z / 2 ** k)
    for i in range(k):
        constrs.append(cp.bmat([[Zs[i], Zs[i + 1]], [Zs[i + 1].T, X]]) >> 0)
    for i in range(m):
        off_diag = -t[i] ** 0.5 * Ts[i]
        constrs.append(cp.bmat([[Zs[k] - X - Ts[i], off_diag], [off_diag.T, X - t[i] * Ts[i]]]) >> 0)
    return (lead_con, constrs)