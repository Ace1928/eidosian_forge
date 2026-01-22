import numpy as np
import scipy as sp
from cvxpy import settings as s
from cvxpy.constraints.exponential import ExpCone as ExpCone_obj
from cvxpy.constraints.nonpos import NonNeg as NonNeg_obj
from cvxpy.constraints.power import PowCone3D as PowCone_obj
from cvxpy.constraints.psd import PSD as PSD_obj
from cvxpy.constraints.second_order import SOC as SOC_obj
from cvxpy.constraints.zero import Zero as Zero_obj
from cvxpy.reductions.solution import Solution
class Slacks:
    """
    CVXPY represents mixed-integer cone programs as

        (Aff)   min{ c.T @ x : A @ x + b in K,
                              x[bools] in {0, 1}, x[ints] in Z } + d.

    Some solvers do not accept input in the form (Aff). A general pattern we find
    across solver types is that the feasible set is represented by

        (Dir)   min{ f @ y : G @ y <=_{K_aff} h, y in K_dir
                             y[bools] in {0, 1}, y[ints] in Z } + d,

    where K_aff is built from a list convex cones which includes the zero cone (ZERO),
    and K_dir is built from a list of convex cones which includes the free cone (FREE).

    This reduction handles mapping back and forth between problems stated in terms
    of (Aff) and (Dir), by way of adding slack variables.

    Notes
    -----
    Support for semidefinite constraints has not yet been implemented in this
    reduction.

    If the problem has no integer constraints, then the Dualize reduction should be
    used instead.

    Because this reduction is only intended for mixed-integer problems, this reduction
    makes no attempt to recover dual variables when mapping between (Aff) and (Dir).
    """

    @staticmethod
    def apply(prob, affine):
        """
        "prob" is a ParamConeProg which represents

            (Aff)   min{ c.T @ x : A @ x + b in K,
                                  x[bools] in {0, 1}, x[ints] in Z } + d.

        We return data for an equivalent problem

            (Dir)   min{ f @ y : G @ y <=_{K_aff} h, y in K_dir
                                 y[bools] in {0, 1}, y[ints] in Z } + d,

        where

            (1) K_aff is built from cone types specified in "affine" (a list of strings),
            (2) a primal solution for (Dir) can be mapped back to a primal solution
                for (Aff) by selecting the leading ``c.size`` block of y's components.

        In the returned dict "data", data[s.A] = G, data[s.B] = h, data[s.C] = f,
        data['K_aff'] = K_aff, data['K_dir'] = K_dir, data[s.BOOL_IDX] = bools,
        and data[s.INT_IDX] = ints. The rows of G are ordered according to ZERO, then
        (as applicable) NONNEG, SOC, and EXP. If  "c" is the objective vector in (Aff),
        then ``y[:c.size]`` should contain the optimal solution to (Aff). The columns of
        G correspond first to variables in cones FREE, then NONNEG, then SOC, then EXP.
        The length of the free cone is equal to ``c.size``.

        Assumptions
        -----------
        The function call ``c, d, A, b = prob.apply_parameters()`` returns (A,b) with
        rows formatted first for the zero cone, then for the nonnegative orthant, then
        second order cones, then the exponential cone. Removing this assumption will
        require adding additional data to ParamConeProg objects.
        """
        c, d, A, b = prob.apply_parameters()
        A = -A
        cone_dims = prob.cone_dims
        if cone_dims.psd:
            raise NotImplementedError()
        for val in affine:
            if val not in {ZERO, NONNEG, EXP, SOC, POW3D}:
                raise NotImplementedError()
        if ZERO not in affine:
            affine.append(ZERO)
        cone_lens = {ZERO: cone_dims.zero, NONNEG: cone_dims.nonneg, SOC: sum(cone_dims.soc), EXP: 3 * cone_dims.exp, POW3D: 3 * len(cone_dims.p3d)}
        row_offsets = {ZERO: 0, NONNEG: cone_lens[ZERO], SOC: cone_lens[ZERO] + cone_lens[NONNEG], EXP: cone_lens[ZERO] + cone_lens[NONNEG] + cone_lens[SOC], POW3D: cone_lens[ZERO] + cone_lens[NONNEG] + cone_lens[SOC] + cone_lens[EXP]}
        A_aff, b_aff = ([], [])
        A_slk, b_slk = ([], [])
        total_slack = 0
        for co_type in [ZERO, NONNEG, SOC, EXP, POW3D]:
            co_dim = cone_lens[co_type]
            if co_dim > 0:
                r = row_offsets[co_type]
                A_temp = A[r:r + co_dim, :]
                b_temp = b[r:r + co_dim]
                if co_type in affine:
                    A_aff.append(A_temp)
                    b_aff.append(b_temp)
                else:
                    total_slack += b_temp.size
                    A_slk.append(A_temp)
                    b_slk.append(b_temp)
        K_dir = {FREE: prob.x.size, NONNEG: 0 if NONNEG in affine else cone_dims.nonneg, SOC: [] if SOC in affine else cone_dims.soc, EXP: 0 if EXP in affine else cone_dims.exp, PSD: [], DUAL_EXP: 0, POW3D: [] if POW3D in affine else cone_dims.p3d, DUAL_POW3D: []}
        K_aff = {NONNEG: cone_dims.nonneg if NONNEG in affine else 0, SOC: cone_dims.soc if SOC in affine else [], EXP: cone_dims.exp if EXP in affine else 0, PSD: [], ZERO: cone_dims.zero + total_slack, POW3D: cone_dims.p3d if POW3D in affine else []}
        data = dict()
        if A_slk:
            A_slk = sp.sparse.vstack(tuple(A_slk))
            eye = sp.sparse.eye(total_slack)
            if A_aff:
                A_aff = sp.sparse.vstack(tuple(A_aff), format='csr')
                G = sp.sparse.bmat([[A_slk, eye], [A_aff, None]])
                h = np.concatenate(b_slk + b_aff)
            else:
                G = sp.sparse.hstack((A_slk, eye))
                h = np.concatenate(b_slk)
            f = np.concatenate((c, np.zeros(total_slack)))
        elif A_aff:
            G = sp.sparse.vstack(tuple(A_aff), format='csr')
            h = np.concatenate(b_aff)
            f = c
        else:
            raise ValueError()
        data[s.A] = G
        data[s.B] = h
        data[s.C] = f
        data[s.BOOL_IDX] = [int(t[0]) for t in prob.x.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in prob.x.integer_idx]
        data['K_dir'] = K_dir
        data['K_aff'] = K_aff
        inv_data = dict()
        inv_data['x_id'] = prob.x.id
        inv_data['K_dir'] = K_dir
        inv_data['K_aff'] = K_aff
        inv_data[s.OBJ_OFFSET] = d
        return (data, inv_data)

    @staticmethod
    def invert(solution, inv_data):
        if solution.status in s.SOLUTION_PRESENT:
            prim_vars = solution.primal_vars
            x = prim_vars[FREE]
            del prim_vars[FREE]
            prim_vars[inv_data['x_id']] = x
        solution.opt_val += inv_data[s.OBJ_OFFSET]
        return solution