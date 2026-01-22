from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
class solution(object):
    """
    Solver solution vectors z, u
    """

    def __init__(self):
        self.x = None
        self.y = None