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
def _solve_as_qp(problem, candidates):
    if _is_lp(problem) and [s for s in candidates['conic_solvers'] if s not in candidates['qp_solvers']]:
        return False
    return candidates['qp_solvers'] and qp2symbolic_qp.accepts(problem)