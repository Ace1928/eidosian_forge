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
def _parse_n_pca_threshold(self, data, n_pca, rank_threshold):
    if isinstance(n_pca, str):
        n_pca = n_pca.lower()
        if n_pca != 'auto':
            raise ValueError("n_pca must be an integer 0 <= n_pca < min(n_samples,n_features), or in [None, False, True, 'auto'].")
    if isinstance(n_pca, numbers.Number):
        if not float(n_pca).is_integer():
            n_pcaR = np.round(n_pca).astype(int)
            warnings.warn('Cannot perform PCA to fractional {} dimensions. Rounding to {}'.format(n_pca, n_pcaR), RuntimeWarning)
            n_pca = n_pcaR
        if n_pca < 0:
            raise ValueError('n_pca cannot be negative. Please supply an integer 0 <= n_pca < min(n_samples,n_features) or None')
        elif np.min(data.shape) <= n_pca:
            warnings.warn('Cannot perform PCA to {} dimensions on data with min(n_samples, n_features) = {}'.format(n_pca, np.min(data.shape)), RuntimeWarning)
            n_pca = 0
    if n_pca in [0, False, None]:
        n_pca = None
    elif n_pca is True:
        n_pca = 'auto'
        _logger.log_info('Estimating n_pca from matrix rank. Supply an integer n_pca for fixed amount.')
    if not any([isinstance(n_pca, numbers.Number), n_pca is None, n_pca == 'auto']):
        raise ValueError('n_pca was not an instance of numbers.Number, could not be cast to False, and not None. Please supply an integer 0 <= n_pca < min(n_samples,n_features) or None')
    if rank_threshold is not None and n_pca != 'auto':
        warnings.warn('n_pca = {}, therefore rank_threshold of {} will not be used. To use rank thresholding, set n_pca = True'.format(n_pca, rank_threshold), RuntimeWarning)
    if n_pca == 'auto':
        if isinstance(rank_threshold, str):
            rank_threshold = rank_threshold.lower()
        if rank_threshold is None:
            rank_threshold = 'auto'
        if isinstance(rank_threshold, numbers.Number):
            if rank_threshold <= 0:
                raise ValueError("rank_threshold must be positive float or 'auto'. ")
        elif rank_threshold != 'auto':
            raise ValueError("rank_threshold must be positive float or 'auto'. ")
    return (n_pca, rank_threshold)