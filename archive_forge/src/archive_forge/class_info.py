from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
class info(object):
    """
    Solver information

    Attributes
    ----------
    iter            - number of iterations taken
    status          - status string, e.g. 'Solved'
    status_val      - status as c_int, defined in constants.h
    status_polish   - polish status: successful (1), not (0)
    obj_val         - primal objective
    pri_res         - norm of primal residual
    dua_res         - norm of dual residual
    setup_time      - time taken for setup phase (seconds)
    solve_time      - time taken for solve phase (seconds)
    update_time     - time taken for update phase (seconds)
    polish_time     - time taken for polish phase (seconds)
    run_time        - total time  (seconds)
    rho_updates     - number of rho updates
    rho_estimate    - optimal rho estimate
    """

    def __init__(self):
        self.iter = 0
        self.status_val = OSQP_UNSOLVED
        self.status = 'Unsolved'
        self.status_polish = 0
        self.update_time = 0.0
        self.polish_time = 0.0
        self.rho_updates = 0.0