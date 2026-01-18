from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_max_iter(self, max_iter_new):
    """
        Update maximum number of iterations
        """
    if max_iter_new <= 0:
        raise ValueError('max_iter must be positive')
    self.work.settings.max_iter = max_iter_new