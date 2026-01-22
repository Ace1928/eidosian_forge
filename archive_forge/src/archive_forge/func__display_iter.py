import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve
from ._optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from ._linprog_util import _postsolve
from ._optimize import OptimizeResult
def _display_iter(phase, iteration, slack, con, fun):
    """
    Print indicators of optimization status to the console.
    """
    header = True if not iteration % 20 else False
    if header:
        print('Phase', 'Iteration', 'Minimum Slack      ', 'Constraint Residual', 'Objective          ')
    fmt = '{0:<6}{1:<10}{2:<20.13}{3:<20.13}{4:<20.13}'
    try:
        slack = np.min(slack)
    except ValueError:
        slack = 'NA'
    print(fmt.format(phase, iteration, slack, np.linalg.norm(con), fun))