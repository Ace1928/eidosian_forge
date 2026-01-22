import numpy as np
from ase.optimize.optimize import Optimizer
from ase.utils.linesearch import LineSearch
class LBFGSLineSearch(LBFGS):
    """This optimizer uses the LBFGS algorithm, but does a line search that
    fulfills the Wolff conditions.
    """

    def __init__(self, *args, **kwargs):
        kwargs['use_line_search'] = True
        LBFGS.__init__(self, *args, **kwargs)