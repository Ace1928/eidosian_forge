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
def _parse_n_pca(X, n_pca):
    if n_pca is not None and n_pca >= min(X.shape):
        return None
    else:
        return n_pca