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
def _construct_chain(self, solver: Optional[str]=None, gp: bool=False, enforce_dpp: bool=False, ignore_dpp: bool=False, canon_backend: str | None=None, solver_opts: Optional[dict]=None) -> SolvingChain:
    """
        Construct the chains required to reformulate and solve the problem.

        In particular, this function

        # finds the candidate solvers
        # constructs the solving chain that performs the
           numeric reductions and solves the problem.

        Arguments
        ---------
        solver : str, optional
            The solver to use. Defaults to ECOS.
        gp : bool, optional
            If True, the problem is parsed as a Disciplined Geometric Program
            instead of as a Disciplined Convex Program.
        enforce_dpp : bool, optional
            Whether to error on DPP violations.
        ignore_dpp : bool, optional
            When True, DPP problems will be treated as non-DPP,
            which may speed up compilation. Defaults to False.
        canon_backend : str, optional
            'CPP' (default) | 'SCIPY'
            Specifies which backend to use for canonicalization, which can affect
            compilation time. Defaults to None, i.e., selecting the default
            backend.
        solver_opts: dict, optional
            Additional arguments to pass to the solver.

        Returns
        -------
        A solving chain
        """
    candidate_solvers = self._find_candidate_solvers(solver=solver, gp=gp)
    self._sort_candidate_solvers(candidate_solvers)
    return construct_solving_chain(self, candidate_solvers, gp=gp, enforce_dpp=enforce_dpp, ignore_dpp=ignore_dpp, canon_backend=canon_backend, solver_opts=solver_opts, specified_solver=solver)