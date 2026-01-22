from . import api
from . import base
from . import graphs
from . import matrix
from . import utils
from functools import partial
from scipy import sparse
import abc
import numpy as np
import pygsp
import tasklogger
@staticmethod
def _parse_n_landmark(X, n_landmark):
    if n_landmark is not None and n_landmark >= X.shape[0]:
        return None
    else:
        return n_landmark