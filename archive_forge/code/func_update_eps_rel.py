from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_eps_rel(self, eps_rel_new):
    """
        Update relative tolerance
        """
    if eps_rel_new <= 0:
        raise ValueError('eps_rel must be positive')
    self.work.settings.eps_rel = eps_rel_new