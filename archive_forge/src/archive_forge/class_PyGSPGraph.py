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
class PyGSPGraph(with_metaclass(abc.ABCMeta, pygsp.graphs.Graph, Base)):
    """Interface between BaseGraph and PyGSP.

    All graphs should possess these matrices. We inherit a lot
    of functionality from pygsp.graphs.Graph.

    There is a lot of overhead involved in having both a weight and
    kernel matrix
    """

    def __init__(self, lap_type='combinatorial', coords=None, plotting=None, **kwargs):
        if plotting is None:
            plotting = {}
        W = self._build_weight_from_kernel(self.K)
        super().__init__(W, lap_type=lap_type, coords=coords, plotting=plotting, **kwargs)

    @property
    @abc.abstractmethod
    def K():
        """Kernel matrix

        Returns
        -------
        K : array-like, shape=[n_samples, n_samples]
            kernel matrix defined as the adjacency matrix with
            ones down the diagonal
        """
        raise NotImplementedError

    def _build_weight_from_kernel(self, kernel):
        """Private method to build an adjacency matrix from
        a kernel matrix

        Just puts zeroes down the diagonal in-place, since the
        kernel matrix is ultimately not stored.

        Parameters
        ----------
        kernel : array-like, shape=[n_samples, n_samples]
            Kernel matrix.

        Returns
        -------
        Adjacency matrix, shape=[n_samples, n_samples]
        """
        weight = kernel.copy()
        self._diagonal = weight.diagonal().copy()
        weight = matrix.set_diagonal(weight, 0)
        return weight