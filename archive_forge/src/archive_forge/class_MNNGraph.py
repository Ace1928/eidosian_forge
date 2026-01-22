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
class MNNGraph(DataGraph):
    """Mutual nearest neighbors graph

    Performs batch correction by forcing connections between batches, but
    only when the connection is mutual (i.e. x is a neighbor of y _and_
    y is a neighbor of x).

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`,
        `scipy.sparse.spmatrix`,
        `pandas.DataFrame`, `pandas.SparseDataFrame`.

    sample_idx : array-like, shape=[n_samples]
        Batch index

    beta : `float`, optional (default: 1)
        Downweight between-batch affinities by beta

    adaptive_k : {'min', 'mean', 'sqrt', `None`} (default: None)
        Weights MNN kernel adaptively using the number of cells in
        each sample according to the selected method.

    Attributes
    ----------
    subgraphs : list of `graphtools.graphs.kNNGraph`
        Graphs representing each batch separately
    """

    def __init__(self, data, sample_idx, knn=5, beta=1, n_pca=None, decay=None, adaptive_k=None, bandwidth=None, distance='euclidean', thresh=0.0001, n_jobs=1, **kwargs):
        self.beta = beta
        self.sample_idx = sample_idx
        self.samples, self.n_cells = np.unique(self.sample_idx, return_counts=True)
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.bandwidth = bandwidth
        self.thresh = thresh
        self.n_jobs = n_jobs
        if sample_idx is None:
            raise ValueError('sample_idx must be given. For a graph without batch correction, use kNNGraph.')
        elif len(sample_idx) != data.shape[0]:
            raise ValueError('sample_idx ({}) must be the same length as data ({})'.format(len(sample_idx), data.shape[0]))
        elif len(self.samples) == 1:
            raise ValueError('sample_idx must contain more than one unique value')
        if adaptive_k is not None:
            warnings.warn('`adaptive_k` has been deprecated. Using fixed knn.', DeprecationWarning)
        super().__init__(data, n_pca=n_pca, **kwargs)

    def _check_symmetrization(self, kernel_symm, theta):
        if (kernel_symm == 'theta' or kernel_symm == 'mnn') and theta is not None and (not isinstance(theta, numbers.Number)):
            raise TypeError('Expected `theta` as a float. Got {}.'.format(type(theta)))
        else:
            super()._check_symmetrization(kernel_symm, theta)

    def get_params(self):
        """Get parameters from this object"""
        params = super().get_params()
        params.update({'beta': self.beta, 'knn': self.knn, 'decay': self.decay, 'bandwidth': self.bandwidth, 'distance': self.distance, 'thresh': self.thresh, 'n_jobs': self.n_jobs})
        return params

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Valid parameters:
        - n_jobs
        - random_state
        - verbose
        Invalid parameters: (these would require modifying the kernel matrix)
        - knn
        - adaptive_k
        - decay
        - distance
        - thresh
        - beta

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        if 'beta' in params and params['beta'] != self.beta:
            raise ValueError('Cannot update beta. Please create a new graph')
        knn_kernel_args = ['knn', 'decay', 'distance', 'thresh', 'bandwidth']
        knn_other_args = ['n_jobs', 'random_state', 'verbose']
        for arg in knn_kernel_args:
            if arg in params and params[arg] != getattr(self, arg):
                raise ValueError('Cannot update {}. Please create a new graph'.format(arg))
        for arg in knn_other_args:
            if arg in params:
                self.__setattr__(arg, params[arg])
                for g in self.subgraphs:
                    g.set_params(**{arg: params[arg]})
        super().set_params(**params)
        return self

    def build_kernel(self):
        """Build the MNN kernel.

        Build a mutual nearest neighbors kernel.

        Returns
        -------
        K : kernel matrix, shape=[n_samples, n_samples]
            symmetric matrix with ones down the diagonal
            with no non-negative entries.
        """
        with _logger.log_task('subgraphs'):
            self.subgraphs = []
            from .api import Graph
            for i, idx in enumerate(self.samples):
                _logger.log_debug('subgraph {}: sample {}, n = {}, knn = {}'.format(i, idx, np.sum(self.sample_idx == idx), self.knn))
                data = self.data_nu[self.sample_idx == idx]
                graph = Graph(data, n_pca=None, knn=self.knn, decay=self.decay, bandwidth=self.bandwidth, distance=self.distance, thresh=self.thresh, verbose=self.verbose, random_state=self.random_state, n_jobs=self.n_jobs, kernel_symm='+', initialize=True)
                self.subgraphs.append(graph)
        with _logger.log_task('MNN kernel'):
            if self.thresh > 0 or self.decay is None:
                K = sparse.lil_matrix((self.data_nu.shape[0], self.data_nu.shape[0]))
            else:
                K = np.zeros([self.data_nu.shape[0], self.data_nu.shape[0]])
            for i, X in enumerate(self.subgraphs):
                K = matrix.set_submatrix(K, self.sample_idx == self.samples[i], self.sample_idx == self.samples[i], X.K)
                within_batch_norm = np.array(np.sum(X.K, 1)).flatten()
                for j, Y in enumerate(self.subgraphs):
                    if i == j:
                        continue
                    with _logger.log_task('kernel from sample {} to {}'.format(self.samples[i], self.samples[j])):
                        Kij = Y.build_kernel_to_data(X.data_nu, knn=self.knn)
                        between_batch_norm = np.array(np.sum(Kij, 1)).flatten()
                        scale = np.minimum(1, within_batch_norm / between_batch_norm) * self.beta
                        if sparse.issparse(Kij):
                            Kij = Kij.multiply(scale[:, None])
                        else:
                            Kij = Kij * scale[:, None]
                        K = matrix.set_submatrix(K, self.sample_idx == self.samples[i], self.sample_idx == self.samples[j], Kij)
        return K

    def build_kernel_to_data(self, Y, theta=None):
        """Build transition matrix from new data to the graph

        Creates a transition matrix such that `Y` can be approximated by
        a linear combination of landmarks. Any
        transformation of the landmarks can be trivially applied to `Y` by
        performing

        `transform_Y = transitions.dot(transform)`

        Parameters
        ----------

        Y : array-like, [n_samples_y, n_features]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        theta : array-like or `None`, optional (default: `None`)
            if `self.theta` is a matrix, theta values must be explicitly
            specified between `Y` and each sample in `self.data`

        Returns
        -------

        transitions : array-like, [n_samples_y, self.data.shape[0]]
            Transition matrix from `Y` to `self.data`
        """
        raise NotImplementedError