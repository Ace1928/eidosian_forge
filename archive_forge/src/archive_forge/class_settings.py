from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
class settings(object):
    """
    OSQP solver settings

    Attributes
    ----------
    -> These cannot be changed without running setup
    sigma    [1e-06]           - Regularization parameter for polish
    scaling  [10]            - Scaling/Equilibration iterations (0 disabled)

    -> These can be changed without running setup
    rho  [1.6]                 - Step in ADMM procedure
    max_iter [4000]                     - Maximum number of iterations
    eps_abs  [1e-05]                    - Absolute tolerance
    eps_rel  [1e-05]                    - Relative tolerance
    eps_prim_inf  [1e-06]                    - Primal infeasibility tolerance
    eps_dual_inf  [1e-06]                    - Dual infeasibility tolerance
    alpha [1.6]                         - Relaxation parameter
    delta [1.0]                         - Regularization parameter for polish
    verbose  [True]                     - Verbosity
    scaled_termination [False]             - Evalute scaled termination criteria
    check_termination  [True]             - Interval for termination checking
    warm_start [False]                  - Reuse solution from previous solve
    polish  [False]                     - Solution polish
    polish_refine_iter  [3]                - Iterative refinement iterations
    """

    def __init__(self, **kwargs):
        self.rho = kwargs.pop('rho', 0.1)
        self.sigma = kwargs.pop('sigma', 1e-06)
        self.scaling = kwargs.pop('scaling', 10)
        self.max_iter = kwargs.pop('max_iter', 4000)
        self.eps_abs = kwargs.pop('eps_abs', 0.001)
        self.eps_rel = kwargs.pop('eps_rel', 0.001)
        self.eps_prim_inf = kwargs.pop('eps_prim_inf', 0.0001)
        self.eps_dual_inf = kwargs.pop('eps_dual_inf', 0.0001)
        self.alpha = kwargs.pop('alpha', 1.6)
        self.linsys_solver = kwargs.pop('linsys_solver', QDLDL_SOLVER)
        self.delta = kwargs.pop('delta', 1e-06)
        self.verbose = kwargs.pop('verbose', True)
        self.scaled_termination = kwargs.pop('scaled_termination', False)
        self.check_termination = kwargs.pop('check_termination', True)
        self.warm_start = kwargs.pop('warm_start', True)
        self.polish = kwargs.pop('polish', False)
        self.polish_refine_iter = kwargs.pop('polish_refine_iter', 3)
        self.adaptive_rho = kwargs.pop('adaptive_rho', True)
        self.adaptive_rho_interval = kwargs.pop('adaptive_rho_interval', 200)
        self.adaptive_rho_tolerance = kwargs.pop('adaptive_rho_tolerance', 5)
        self.adaptive_rho_fraction = kwargs.pop('adaptive_rho_fraction', 0.7)