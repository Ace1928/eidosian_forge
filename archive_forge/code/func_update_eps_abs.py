from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_eps_abs(self, eps_abs_new):
    """
        Update absolute tolerance
        """
    if eps_abs_new <= 0:
        raise ValueError('eps_abs must be positive')
    self.work.settings.eps_abs = eps_abs_new