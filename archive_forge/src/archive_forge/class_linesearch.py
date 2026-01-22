from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
class linesearch(object):
    """
    Vectors obtained from line search between the ADMM and the polished
    solution

    Attributes
    ----------
    X     - matrix in R^{N \\times n}
    Z     - matrix in R^{N \\times m}
    Y     - matrix in R^{N \\times m}
    t     - vector in R^N
    """

    def __init__(self):
        self.X = None
        self.Z = None
        self.Y = None
        self.t = None