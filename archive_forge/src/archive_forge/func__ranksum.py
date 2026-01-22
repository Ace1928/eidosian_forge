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
def _ranksum(X, sum_idx, axis=0):
    X = utils.to_array_or_spmatrix(X)
    if sparse.issparse(X):
        ranksums = []
        if axis == 1:
            next_fn = X.getrow
        elif axis == 0:
            next_fn = X.getcol
        for i in range(X.shape[(axis + 1) % 2]):
            data = next_fn(i)
            rank = _rank(data, axis=axis)
            ranksums.append(np.sum(rank[sum_idx]))
        return np.array(ranksums)
    else:
        data_rank = _rank(X, axis=0)
        return np.sum(data_rank[sum_idx], axis=0)