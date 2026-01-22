from . import plot
from . import select
from . import utils
from ._lazyload import matplotlib
from scipy import sparse
from scipy import stats
from sklearn import metrics
from sklearn import neighbors
import joblib
import numbers
import numpy as np
import pandas as pd
import warnings
def _vector_coerce_dense(x):
    x = utils.toarray(x)
    x_1d = x.flatten()
    if not len(x_1d) == x.shape[0]:
        raise ValueError('x must be a 1D array. Got shape {}'.format(x.shape))
    return x_1d