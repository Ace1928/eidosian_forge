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
class DataGraph(with_metaclass(abc.ABCMeta, Data, BaseGraph)):
    """Abstract class for graphs built from a dataset

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`.

    n_pca : {`int`, `None`, `bool`, 'auto'}, optional (default: `None`)
        number of PC dimensions to retain for graph building.
        If n_pca in `[None,False,0]`, uses the original data.
        If `True` then estimate using a singular value threshold
        Note: if data is sparse, uses SVD instead of PCA
        TODO: should we subtract and store the mean?

    rank_threshold : `float`, 'auto', optional (default: 'auto')
        threshold to use when estimating rank for
        `n_pca in [True, 'auto']`.
        Note that the default kwarg is `None` for this parameter.
        It is subsequently parsed to 'auto' if necessary.
        If 'auto', this threshold is
        smax * np.finfo(data.dtype).eps * max(data.shape)
        where smax is the maximum singular value of the data matrix.
        For reference, see, e.g.
        W. Press, S. Teukolsky, W. Vetterling and B. Flannery,
        “Numerical Recipes (3rd edition)”,
        Cambridge University Press, 2007, page 795.

    random_state : `int` or `None`, optional (default: `None`)
        Random state for random PCA and graph building

    verbose : `bool`, optional (default: `True`)
        Verbosity.

    n_jobs : `int`, optional (default : 1)
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used
    """

    def __init__(self, data, verbose=True, n_jobs=1, **kwargs):
        self.n_jobs = n_jobs
        self.verbose = verbose
        _logger.set_level(verbose)
        super().__init__(data, **kwargs)

    def get_params(self):
        """Get parameters from this object"""
        params = Data.get_params(self)
        params.update(BaseGraph.get_params(self))
        return params

    @abc.abstractmethod
    def build_kernel_to_data(self, Y):
        """Build a kernel from new input data `Y` to the `self.data`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        K_yx: array-like, [n_samples_y, n_samples]
            kernel matrix where each row represents affinities of a single
            sample in `Y` to all samples in `self.data`.

        Raises
        ------

        ValueError: if this Graph is not capable of extension or
        if the supplied data is the wrong shape
        """
        raise NotImplementedError

    def _check_extension_shape(self, Y):
        """Private method to check if new data matches `self.data`

        Parameters
        ----------
        Y : array-like, shape=[n_samples_y, n_features_y]
            Input data

        Returns
        -------
        Y : array-like, shape=[n_samples_y, n_pca]
            (Potentially transformed) input data

        Raises
        ------
        ValueError : if `n_features_y` is not either `self.data.shape[1]` or
        `self.n_pca`.
        """
        if len(Y.shape) != 2:
            raise ValueError('Expected a 2D matrix. Y has shape {}'.format(Y.shape))
        if not Y.shape[1] == self.data_nu.shape[1]:
            if Y.shape[1] == self.data.shape[1]:
                Y = self.transform(Y)
            else:
                if self.data.shape[1] != self.data_nu.shape[1]:
                    msg = 'Y must be of shape either (n, {}) or (n, {})'.format(self.data.shape[1], self.data_nu.shape[1])
                else:
                    msg = 'Y must be of shape (n, {})'.format(self.data.shape[1])
                raise ValueError(msg)
        return Y

    def extend_to_data(self, Y):
        """Build transition matrix from new data to the graph

        Creates a transition matrix such that `Y` can be approximated by
        a linear combination of samples in `self.data`. Any
        transformation of `self.data` can be trivially applied to `Y` by
        performing

        `transform_Y = self.interpolate(transform, transitions)`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        transitions : array-like, shape=[n_samples_y, self.data.shape[0]]
            Transition matrix from `Y` to `self.data`
        """
        Y = self._check_extension_shape(Y)
        kernel = self.build_kernel_to_data(Y)
        transitions = normalize(kernel, norm='l1', axis=1)
        return transitions

    def interpolate(self, transform, transitions=None, Y=None):
        """Interpolate new data onto a transformation of the graph data

        One of either transitions or Y should be provided

        Parameters
        ----------

        transform : array-like, shape=[n_samples, n_transform_features]

        transitions : array-like, optional, shape=[n_samples_y, n_samples]
            Transition matrix from `Y` (not provided) to `self.data`

        Y: array-like, optional, shape=[n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        Y_transform : array-like, [n_samples_y, n_features or n_pca]
            Transition matrix from `Y` to `self.data`

        Raises
        ------
        ValueError: if neither `transitions` nor `Y` is provided
        """
        if transitions is None:
            if Y is None:
                raise ValueError('Either `transitions` or `Y` must be provided.')
            else:
                transitions = self.extend_to_data(Y)
        Y_transform = transitions.dot(transform)
        return Y_transform

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Valid parameters:
        - n_jobs
        - verbose

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        if 'n_jobs' in params:
            self.n_jobs = params['n_jobs']
        if 'verbose' in params:
            self.verbose = params['verbose']
            _logger.set_level(self.verbose)
        super().set_params(**params)
        return self