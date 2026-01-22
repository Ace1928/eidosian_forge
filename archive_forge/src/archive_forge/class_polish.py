from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
class polish(object):
    """
    Polishing structure containing active constraints at the solution

    Attributes
    ----------
    ind_low         - indices of lower-active constraints
    ind_upp         - indices of upper-active constraints
    n_low           - number of lower-active constraints
    n_upp           - number of upper-active constraints
    Ared            - Part of A containing only active rows
    x               - polished x
    z               - polished z
    y               - polished y
    """

    def __init__(self):
        self.ind_low = None
        self.ind_upp = None
        self.n_low = None
        self.n_upp = None
        self.Ared = None
        self.x = None
        self.z = None
        self.y = None