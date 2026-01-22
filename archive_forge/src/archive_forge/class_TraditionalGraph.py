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
class TraditionalGraph(DataGraph):
    """Traditional weighted adjacency graph

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`,
        `pandas.DataFrame`, `pandas.SparseDataFrame`.
        If `precomputed` is not `None`, data should be an
        [n_samples, n_samples] matrix denoting pairwise distances,
        affinities, or edge weights.

    knn : `int`, optional (default: 5)
        Number of nearest neighbors (including self) to use to build the graph

    decay : `int` or `None`, optional (default: 40)
        Rate of alpha decay to use. If `None`, alpha decay is not used.

    bandwidth : `float`, list-like,`callable`, or `None`, optional (default: `None`)
        Fixed bandwidth to use. If given, overrides `knn`. Can be a single
        bandwidth, list-like (shape=[n_samples]) of bandwidths for each
        sample, or a `callable` that takes in a `n x m` matrix and returns a
        a single value or list-like of length n (shape=[n_samples])

    bandwidth_scale : `float`, optional (default : 1.0)
        Rescaling factor for bandwidth.

    distance : `str`, optional (default: `'euclidean'`)
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph.
        TODO: actually sklearn.neighbors has even more choices

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

    thresh : `float`, optional (default: `1e-4`)
        Threshold above which to calculate alpha decay kernel.
        All affinities below `thresh` will be set to zero in order to save
        on time and memory constraints.

    precomputed : {'distance', 'affinity', 'adjacency', `None`},
        optional (default: `None`)
        If the graph is precomputed, this variable denotes which graph
        matrix is provided as `data`.
        Only one of `precomputed` and `n_pca` can be set.
    """

    def __init__(self, data, knn=5, decay=40, bandwidth=None, bandwidth_scale=1.0, distance='euclidean', n_pca=None, thresh=0.0001, precomputed=None, **kwargs):
        if decay is None and precomputed not in ['affinity', 'adjacency']:
            raise ValueError('`decay` must be provided for a TraditionalGraph. For kNN kernel, use kNNGraph.')
        if precomputed is not None and n_pca not in [None, 0, False]:
            n_pca = None
            warnings.warn('n_pca cannot be given on a precomputed graph. Setting n_pca=None', RuntimeWarning)
        if knn is None and bandwidth is None:
            raise ValueError('Either `knn` or `bandwidth` must be provided.')
        if knn is not None and knn > data.shape[0] - 2:
            warnings.warn('Cannot set knn ({k}) to be greater than  n_samples - 2 ({n}). Setting knn={n}'.format(k=knn, n=data.shape[0] - 2))
            knn = data.shape[0] - 2
        if precomputed is not None:
            if precomputed not in ['distance', 'affinity', 'adjacency']:
                raise ValueError("Precomputed value {} not recognized. Choose from ['distance', 'affinity', 'adjacency']".format(precomputed))
            elif data.shape[0] != data.shape[1]:
                raise ValueError('Precomputed {} must be a square matrix. {} was given'.format(precomputed, data.shape))
            elif (data < 0).sum() > 0:
                raise ValueError('Precomputed {} should be non-negative'.format(precomputed))
        self.knn = knn
        self.decay = decay
        self.bandwidth = bandwidth
        self.bandwidth_scale = bandwidth_scale
        self.distance = distance
        self.thresh = thresh
        self.precomputed = precomputed
        super().__init__(data, n_pca=n_pca, **kwargs)

    def get_params(self):
        """Get parameters from this object"""
        params = super().get_params()
        params.update({'knn': self.knn, 'decay': self.decay, 'bandwidth': self.bandwidth, 'bandwidth_scale': self.bandwidth_scale, 'distance': self.distance, 'precomputed': self.precomputed})
        return params

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Invalid parameters: (these would require modifying the kernel matrix)
        - precomputed
        - distance
        - knn
        - decay
        - bandwidth
        - bandwidth_scale

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        if 'precomputed' in params and params['precomputed'] != self.precomputed:
            raise ValueError('Cannot update precomputed. Please create a new graph')
        if 'distance' in params and params['distance'] != self.distance and (self.precomputed is None):
            raise ValueError('Cannot update distance. Please create a new graph')
        if 'knn' in params and params['knn'] != self.knn and (self.precomputed is None):
            raise ValueError('Cannot update knn. Please create a new graph')
        if 'decay' in params and params['decay'] != self.decay and (self.precomputed is None):
            raise ValueError('Cannot update decay. Please create a new graph')
        if 'bandwidth' in params and params['bandwidth'] != self.bandwidth and (self.precomputed is None):
            raise ValueError('Cannot update bandwidth. Please create a new graph')
        if 'bandwidth_scale' in params and params['bandwidth_scale'] != self.bandwidth_scale:
            raise ValueError('Cannot update bandwidth_scale. Please create a new graph')
        super().set_params(**params)
        return self

    def build_kernel(self):
        """Build the KNN kernel.

        Build a k nearest neighbors kernel, optionally with alpha decay.
        If `precomputed` is not `None`, the appropriate steps in the kernel
        building process are skipped.
        Must return a symmetric matrix

        Returns
        -------
        K : kernel matrix, shape=[n_samples, n_samples]
            symmetric matrix with ones down the diagonal
            with no non-negative entries.

        Raises
        ------
        ValueError: if `precomputed` is not an acceptable value
        """
        if self.precomputed == 'affinity':
            K = self.data_nu
        elif self.precomputed == 'adjacency':
            K = self.data_nu
            if sparse.issparse(K) and (not (isinstance(K, sparse.dok_matrix) or isinstance(K, sparse.lil_matrix))):
                K = K.tolil()
            K = matrix.set_diagonal(K, 1)
        else:
            with _logger.log_task('affinities'):
                if sparse.issparse(self.data_nu):
                    self.data_nu = self.data_nu.toarray()
                if self.precomputed == 'distance':
                    pdx = self.data_nu
                elif self.precomputed is None:
                    pdx = pdist(self.data_nu, metric=self.distance)
                    if np.any(pdx == 0):
                        pdx = squareform(pdx)
                        duplicate_ids = np.array([i for i in np.argwhere(pdx == 0) if i[1] > i[0]])
                        if len(duplicate_ids) < 20:
                            duplicate_names = ', '.join(['{} and {}'.format(i[0], i[1]) for i in duplicate_ids])
                            warnings.warn('Detected zero distance between samples {}. Consider removing duplicates to avoid errors in downstream processing.'.format(duplicate_names), RuntimeWarning)
                        else:
                            warnings.warn('Detected zero distance between {} pairs of samples. Consider removing duplicates to avoid errors in downstream processing.'.format(len(duplicate_ids)), RuntimeWarning)
                    else:
                        pdx = squareform(pdx)
                else:
                    raise ValueError("precomputed='{}' not recognized. Choose from ['affinity', 'adjacency', 'distance', None]".format(self.precomputed))
                if self.bandwidth is None:
                    knn_dist = np.partition(pdx, self.knn + 1, axis=1)[:, :self.knn + 1]
                    bandwidth = np.max(knn_dist, axis=1)
                elif callable(self.bandwidth):
                    bandwidth = self.bandwidth(pdx)
                else:
                    bandwidth = self.bandwidth
                bandwidth = bandwidth * self.bandwidth_scale
                pdx = (pdx.T / bandwidth).T
                K = np.exp(-1 * np.power(pdx, self.decay))
                K = np.where(np.isnan(K), 1, K)
        if sparse.issparse(K):
            if not (isinstance(K, sparse.csr_matrix) or isinstance(K, sparse.csc_matrix) or isinstance(K, sparse.bsr_matrix)):
                K = K.tocsr()
            K.data[K.data < self.thresh] = 0
            K = K.tocoo()
            K.eliminate_zeros()
            K = K.tocsr()
        else:
            K[K < self.thresh] = 0
        return K

    def build_kernel_to_data(self, Y, knn=None, bandwidth=None, bandwidth_scale=None):
        """Build transition matrix from new data to the graph

        Creates a transition matrix such that `Y` can be approximated by
        a linear combination of landmarks. Any
        transformation of the landmarks can be trivially applied to `Y` by
        performing

        `transform_Y = transitions.dot(transform)`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_features]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        transitions : array-like, [n_samples_y, self.data.shape[0]]
            Transition matrix from `Y` to `self.data`

        Raises
        ------

        ValueError: if `precomputed` is not `None`, then the graph cannot
        be extended.
        """
        if knn is None:
            knn = self.knn
        if bandwidth is None:
            bandwidth = self.bandwidth
        if bandwidth_scale is None:
            bandwidth_scale = self.bandwidth_scale
        if self.precomputed is not None:
            raise ValueError('Cannot extend kernel on precomputed graph')
        else:
            with _logger.log_task('affinities'):
                Y = self._check_extension_shape(Y)
                pdx = cdist(Y, self.data_nu, metric=self.distance)
                if bandwidth is None:
                    knn_dist = np.partition(pdx, knn, axis=1)[:, :knn]
                    bandwidth = np.max(knn_dist, axis=1)
                elif callable(bandwidth):
                    bandwidth = bandwidth(pdx)
                bandwidth = bandwidth_scale * bandwidth
                pdx = (pdx.T / bandwidth).T
                K = np.exp(-1 * pdx ** self.decay)
                K = np.where(np.isnan(K), 1, K)
                K[K < self.thresh] = 0
        return K

    @property
    def weighted(self):
        if self.precomputed is not None:
            return not matrix.nonzero_discrete(self.K, [0.5, 1])
        else:
            return super().weighted

    def _check_shortest_path_distance(self, distance):
        if self.precomputed is not None:
            if distance == 'data':
                raise ValueError("Graph shortest path with data distance not valid for precomputed graphs. For precomputed graphs, use `distance='constant'` for unweighted graphs and `distance='affinity'` for weighted graphs.")
        super()._check_shortest_path_distance(distance)

    def _default_shortest_path_distance(self):
        if self.precomputed is not None and (not self.weighted):
            distance = 'constant'
            _logger.log_info('Using constant distances.')
        else:
            distance = super()._default_shortest_path_distance()
        return distance