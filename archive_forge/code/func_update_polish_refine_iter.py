from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_polish_refine_iter(self, polish_refine_iter_new):
    """
        Update number iterative refinement iterations in polish
        """
    if polish_refine_iter_new < 0:
        raise ValueError('polish_refine_iter must be nonnegative')
    self.work.settings.polish_refine_iter = polish_refine_iter_new