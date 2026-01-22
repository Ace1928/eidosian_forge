from __future__ import division
from . import matrix
from . import utils
from .base import DataGraph
from .base import PyGSPGraph
from builtins import super
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
import numbers
import numpy as np
import tasklogger
import warnings
def _landmarks_to_data(self):
    landmarks = np.unique(self.clusters)
    if sparse.issparse(self.kernel):
        pmn = sparse.vstack([sparse.csr_matrix(self.kernel[self.clusters == i, :].sum(axis=0)) for i in landmarks])
    else:
        pmn = np.array([np.sum(self.kernel[self.clusters == i, :], axis=0) for i in landmarks])
    return pmn