from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_verbose(self, verbose_new):
    """
        Update verbose parameter
        """
    if (verbose_new is not True) & (verbose_new is not False):
        raise ValueError('verbose should be either True or False')
    self.work.settings.verbose = verbose_new