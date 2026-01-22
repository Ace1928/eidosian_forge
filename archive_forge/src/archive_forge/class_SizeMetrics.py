from __future__ import annotations
import time
import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import cvxpy.utilities as u
import cvxpy.utilities.performance_utils as perf
from cvxpy import Constant, error
from cvxpy import settings as s
from cvxpy.atoms.atom import Atom
from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from cvxpy.constraints.constraint import Constraint
from cvxpy.error import DPPError
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.variable import Variable
from cvxpy.interface.matrix_utilities import scalar_value
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.reductions import InverseData
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.dgp2dcp.dgp2dcp import Dgp2Dcp
from cvxpy.reductions.dqcp2dcp import dqcp2dcp
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solution import INF_OR_UNB_MESSAGE
from cvxpy.reductions.solvers import bisection
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import SOLVER_MAP_CONIC, SOLVER_MAP_QP
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.solvers.solving_chain import (
from cvxpy.settings import SOLVERS
from cvxpy.utilities import debug_tools
from cvxpy.utilities.deterministic import unique_list
class SizeMetrics:
    """Reports various metrics regarding the problem.

    Attributes
    ----------

    num_scalar_variables : integer
        The number of scalar variables in the problem.
    num_scalar_data : integer
        The number of scalar constants and parameters in the problem. The number of
        constants used across all matrices, vectors, in the problem.
        Some constants are not apparent when the problem is constructed: for example,
        The sum_squares expression is a wrapper for a quad_over_lin expression with a
        constant 1 in the denominator.
    num_scalar_eq_constr : integer
        The number of scalar equality constraints in the problem.
    num_scalar_leq_constr : integer
        The number of scalar inequality constraints in the problem.

    max_data_dimension : integer
        The longest dimension of any data block constraint or parameter.
    max_big_small_squared : integer
        The maximum value of (big)(small)^2 over all data blocks of the problem, where
        (big) is the larger dimension and (small) is the smaller dimension
        for each data block.
    """

    def __init__(self, problem: Problem) -> None:
        self.num_scalar_variables = 0
        for var in problem.variables():
            self.num_scalar_variables += var.size
        self.max_data_dimension = 0
        self.num_scalar_data = 0
        self.max_big_small_squared = 0
        for const in problem.constants() + problem.parameters():
            big = 0
            self.num_scalar_data += const.size
            big = 1 if len(const.shape) == 0 else max(const.shape)
            small = 1 if len(const.shape) == 0 else min(const.shape)
            if self.max_data_dimension < big:
                self.max_data_dimension = big
            max_big_small_squared = float(big) * float(small) ** 2
            if self.max_big_small_squared < max_big_small_squared:
                self.max_big_small_squared = max_big_small_squared
        self.num_scalar_eq_constr = 0
        for constraint in problem.constraints:
            if isinstance(constraint, (Equality, Zero)):
                self.num_scalar_eq_constr += constraint.expr.size
        self.num_scalar_leq_constr = 0
        for constraint in problem.constraints:
            if isinstance(constraint, (Inequality, NonPos, NonNeg)):
                self.num_scalar_leq_constr += constraint.expr.size