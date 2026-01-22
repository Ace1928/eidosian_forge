from . import matrix
from . import utils
from builtins import super
from copy import copy as shallow_copy
from future.utils import with_metaclass
from inspect import signature
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import abc
import numbers
import numpy as np
import pickle
import pygsp
import sys
import tasklogger
import warnings
def _reduce_data(self):
    """Private method to reduce data dimension.

        If data is dense, uses randomized PCA. If data is sparse, uses
        randomized SVD.
        TODO: should we subtract and store the mean?
        TODO: Fix the rank estimation so we do not compute the full SVD.

        Returns
        -------
        Reduced data matrix
        """
    if self.n_pca is not None and (self.n_pca == 'auto' or self.n_pca < self.data.shape[1]):
        with _logger.log_task('PCA'):
            n_pca = self.data.shape[1] - 1 if self.n_pca == 'auto' else self.n_pca
            if sparse.issparse(self.data):
                if isinstance(self.data, sparse.coo_matrix) or isinstance(self.data, sparse.lil_matrix) or isinstance(self.data, sparse.dok_matrix):
                    self.data = self.data.tocsr()
                self.data_pca = TruncatedSVD(n_pca, random_state=self.random_state)
            else:
                self.data_pca = PCA(n_pca, svd_solver='randomized', random_state=self.random_state)
            self.data_pca.fit(self.data)
            if self.n_pca == 'auto':
                s = self.data_pca.singular_values_
                smax = s.max()
                if self.rank_threshold == 'auto':
                    threshold = smax * np.finfo(self.data.dtype).eps * max(self.data.shape)
                    self.rank_threshold = threshold
                threshold = self.rank_threshold
                gate = np.where(s >= threshold)[0]
                self.n_pca = gate.shape[0]
                if self.n_pca == 0:
                    raise ValueError('Supplied threshold {} was greater than maximum singular value {} for the data matrix'.format(threshold, smax))
                _logger.log_info('Using rank estimate of {} as n_pca'.format(self.n_pca))
                op = self.data_pca
                op.components_ = op.components_[gate, :]
                op.explained_variance_ = op.explained_variance_[gate]
                op.explained_variance_ratio_ = op.explained_variance_ratio_[gate]
                op.singular_values_ = op.singular_values_[gate]
                self.data_pca = op
            data_nu = self.data_pca.transform(self.data)
        return data_nu
    else:
        data_nu = self.data
        if sparse.issparse(data_nu) and (not isinstance(data_nu, (sparse.csr_matrix, sparse.csc_matrix, sparse.bsr_matrix))):
            data_nu = data_nu.tocsr()
        return data_nu