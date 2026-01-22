from __future__ import annotations
import warnings
import numpy as np
from cvxpy.atoms import EXP_ATOMS, NONPOS_ATOMS, PSD_ATOMS, SOC_ATOMS
from cvxpy.constraints import (
from cvxpy.constraints.exponential import OpRelEntrConeQuad, RelEntrConeQuad
from cvxpy.error import DCPError, DGPError, DPPError, SolverError
from cvxpy.problems.objective import Maximize
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.complex2real import complex2real
from cvxpy.reductions.cone2cone.approximations import APPROX_CONES, QuadApprox
from cvxpy.reductions.cone2cone.exotic2common import (
from cvxpy.reductions.cone2cone.soc2psd import SOC2PSD
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.dgp2dcp.dgp2dcp import Dgp2Dcp
from cvxpy.reductions.discrete2mixedint.valinvec2mixedint import (
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.qp2quad_form import qp2symbolic_qp
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.reductions.solvers.constant_solver import ConstantSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.settings import ECOS, PARAM_THRESHOLD
from cvxpy.utilities.debug_tools import build_non_disciplined_error_msg
def _reductions_for_problem_class(problem, candidates, gp: bool=False, solver_opts=None) -> list[Reduction]:
    """
    Builds a chain that rewrites a problem into an intermediate
    representation suitable for numeric reductions.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    candidates : dict
        Dictionary of candidate solvers divided in qp_solvers
        and conic_solvers.
    gp : bool
        If True, the problem is parsed as a Disciplined Geometric Program
        instead of as a Disciplined Convex Program.
    Returns
    -------
    list of Reduction objects
        A list of reductions that can be used to convert the problem to an
        intermediate form.
    Raises
    ------
    DCPError
        Raised if the problem is not DCP and `gp` is False.
    DGPError
        Raised if the problem is not DGP and `gp` is True.
    """
    reductions = []
    if complex2real.accepts(problem):
        reductions += [complex2real.Complex2Real()]
    if gp:
        reductions += [Dgp2Dcp()]
    if not gp and (not problem.is_dcp()):
        append = build_non_disciplined_error_msg(problem, 'DCP')
        if problem.is_dgp():
            append += '\nHowever, the problem does follow DGP rules. Consider calling solve() with `gp=True`.'
        elif problem.is_dqcp():
            append += '\nHowever, the problem does follow DQCP rules. Consider calling solve() with `qcp=True`.'
        raise DCPError('Problem does not follow DCP rules. Specifically:\n' + append)
    elif gp and (not problem.is_dgp()):
        append = build_non_disciplined_error_msg(problem, 'DGP')
        if problem.is_dcp():
            append += '\nHowever, the problem does follow DCP rules. Consider calling solve() with `gp=False`.'
        elif problem.is_dqcp():
            append += '\nHowever, the problem does follow DQCP rules. Consider calling solve() with `qcp=True`.'
        raise DGPError('Problem does not follow DGP rules.' + append)
    if type(problem.objective) == Maximize:
        reductions += [FlipObjective()]
    use_quad = True if solver_opts is None else solver_opts.get('use_quad_obj', True)
    if _solve_as_qp(problem, candidates) and use_quad:
        reductions += [CvxAttr2Constr(), qp2symbolic_qp.Qp2SymbolicQp()]
    elif not candidates['conic_solvers']:
        raise SolverError('Problem could not be reduced to a QP, and no conic solvers exist among candidate solvers (%s).' % candidates)
    constr_types = {type(c) for c in problem.constraints}
    if FiniteSet in constr_types:
        reductions += [Valinvec2mixedint()]
    return reductions