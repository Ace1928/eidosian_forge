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
class kNNGraph(DataGraph):
    """
    K nearest neighbors graph

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`,
        `pandas.DataFrame`, `pandas.SparseDataFrame`.

    knn : `int`, optional (default: 5)
        Number of nearest neighbors (including self) to use to build the graph

    decay : `int` or `None`, optional (default: `None`)
        Rate of alpha decay to use. If `None`, alpha decay is not used.

    bandwidth : `float`, list-like,`callable`, or `None`,
                optional (default: `None`)
        Fixed bandwidth to use. If given, overrides `knn`. Can be a single
        bandwidth, or a list-like (shape=[n_samples]) of bandwidths for each
        sample

    bandwidth_scale : `float`, optional (default : 1.0)
        Rescaling factor for bandwidth.

    distance : `str`, optional (default: `'euclidean'`)
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph. Custom distance
        functions of form `f(x, y) = d` are also accepted.
        TODO: actually sklearn.neighbors has even more choices

    thresh : `float`, optional (default: `1e-4`)
        Threshold above which to calculate alpha decay kernel.
        All affinities below `thresh` will be set to zero in order to save
        on time and memory constraints.

    Attributes
    ----------

    knn_tree : `sklearn.neighbors.NearestNeighbors`
        The fitted KNN tree. (cached)
        TODO: can we be more clever than sklearn when it comes to choosing
        between KD tree, ball tree and brute force?
    """

    def __init__(self, data, knn=5, decay=None, knn_max=None, search_multiplier=6, bandwidth=None, bandwidth_scale=1.0, distance='euclidean', thresh=0.0001, n_pca=None, **kwargs):
        if decay is not None:
            if thresh <= 0 and knn_max is None:
                raise ValueError('Cannot instantiate a kNNGraph with `decay=None`, `thresh=0` and `knn_max=None`. Use a TraditionalGraph instead.')
            elif thresh < np.finfo(float).eps:
                thresh = np.finfo(float).eps
        if callable(bandwidth):
            raise NotImplementedError('Callable bandwidth is only supported by graphtools.graphs.TraditionalGraph.')
        if knn is None and bandwidth is None:
            raise ValueError('Either `knn` or `bandwidth` must be provided.')
        elif knn is None and bandwidth is not None:
            knn = 5
        if decay is None and bandwidth is not None:
            warnings.warn('`bandwidth` is not used when `decay=None`.', UserWarning)
        if knn > data.shape[0] - 2:
            warnings.warn('Cannot set knn ({k}) to be greater than n_samples - 2 ({n}). Setting knn={n}'.format(k=knn, n=data.shape[0] - 2))
            knn = data.shape[0] - 2
        if knn_max is not None and knn_max < knn:
            warnings.warn('Cannot set knn_max ({knn_max}) to be less than knn ({knn}). Setting knn_max={knn}'.format(knn=knn, knn_max=knn_max))
            knn_max = knn
        if n_pca in [None, 0, False] and data.shape[1] > 500:
            warnings.warn('Building a kNNGraph on data of shape {} is expensive. Consider setting n_pca.'.format(data.shape), UserWarning)
        self.knn = knn
        self.knn_max = knn_max
        self.search_multiplier = search_multiplier
        self.decay = decay
        self.bandwidth = bandwidth
        self.bandwidth_scale = bandwidth_scale
        self.distance = distance
        self.thresh = thresh
        super().__init__(data, n_pca=n_pca, **kwargs)

    def get_params(self):
        """Get parameters from this object"""
        params = super().get_params()
        params.update({'knn': self.knn, 'decay': self.decay, 'bandwidth': self.bandwidth, 'bandwidth_scale': self.bandwidth_scale, 'knn_max': self.knn_max, 'distance': self.distance, 'thresh': self.thresh, 'n_jobs': self.n_jobs, 'random_state': self.random_state, 'verbose': self.verbose})
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
        - knn_max
        - decay
        - bandwidth
        - bandwidth_scale
        - distance
        - thresh

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        if 'knn' in params and params['knn'] != self.knn:
            raise ValueError('Cannot update knn. Please create a new graph')
        if 'knn_max' in params and params['knn_max'] != self.knn:
            raise ValueError('Cannot update knn_max. Please create a new graph')
        if 'decay' in params and params['decay'] != self.decay:
            raise ValueError('Cannot update decay. Please create a new graph')
        if 'bandwidth' in params and params['bandwidth'] != self.bandwidth:
            raise ValueError('Cannot update bandwidth. Please create a new graph')
        if 'bandwidth_scale' in params and params['bandwidth_scale'] != self.bandwidth_scale:
            raise ValueError('Cannot update bandwidth_scale. Please create a new graph')
        if 'distance' in params and params['distance'] != self.distance:
            raise ValueError('Cannot update distance. Please create a new graph')
        if 'thresh' in params and params['thresh'] != self.thresh and (self.decay != 0):
            raise ValueError('Cannot update thresh. Please create a new graph')
        if 'n_jobs' in params:
            self.n_jobs = params['n_jobs']
            if hasattr(self, '_knn_tree'):
                self.knn_tree.set_params(n_jobs=self.n_jobs)
        if 'random_state' in params:
            self.random_state = params['random_state']
        if 'verbose' in params:
            self.verbose = params['verbose']
        super().set_params(**params)
        return self

    @property
    def knn_tree(self):
        """KNN tree object (cached)

        Builds or returns the fitted KNN tree.
        TODO: can we be more clever than sklearn when it comes to choosing
        between KD tree, ball tree and brute force?

        Returns
        -------
        knn_tree : `sklearn.neighbors.NearestNeighbors`
        """
        try:
            return self._knn_tree
        except AttributeError:
            try:
                self._knn_tree = NearestNeighbors(n_neighbors=self.knn + 1, algorithm='ball_tree', metric=self.distance, n_jobs=self.n_jobs).fit(self.data_nu)
            except ValueError:
                warnings.warn('Metric {} not valid for `sklearn.neighbors.BallTree`. Graph instantiation may be slower than normal.'.format(self.distance), UserWarning)
                self._knn_tree = NearestNeighbors(n_neighbors=self.knn + 1, algorithm='auto', metric=self.distance, n_jobs=self.n_jobs).fit(self.data_nu)
            return self._knn_tree

    def build_kernel(self):
        """Build the KNN kernel.

        Build a k nearest neighbors kernel, optionally with alpha decay.
        Must return a symmetric matrix

        Returns
        -------
        K : kernel matrix, shape=[n_samples, n_samples]
            symmetric matrix with ones down the diagonal
            with no non-negative entries.
        """
        knn_max = self.knn_max + 1 if self.knn_max else None
        K = self.build_kernel_to_data(self.data_nu, knn=self.knn + 1, knn_max=knn_max)
        return K

    def _check_duplicates(self, distances, indices):
        if np.any(distances[:, 1] == 0):
            has_duplicates = distances[:, 1] == 0
            if np.sum(distances[:, 1:] == 0) < 20:
                idx = np.argwhere((distances == 0) & has_duplicates[:, None])
                duplicate_ids = np.array([[indices[i[0], i[1]], i[0]] for i in idx if indices[i[0], i[1]] < i[0]])
                duplicate_ids = duplicate_ids[np.argsort(duplicate_ids[:, 0])]
                duplicate_names = ', '.join(['{} and {}'.format(i[0], i[1]) for i in duplicate_ids])
                warnings.warn('Detected zero distance between samples {}. Consider removing duplicates to avoid errors in downstream processing.'.format(duplicate_names), RuntimeWarning)
            else:
                warnings.warn('Detected zero distance between {} pairs of samples. Consider removing duplicates to avoid errors in downstream processing.'.format(np.sum(np.sum(distances[:, 1:] == 0)) // 2), RuntimeWarning)

    def build_kernel_to_data(self, Y, knn=None, knn_max=None, bandwidth=None, bandwidth_scale=None):
        """Build a kernel from new input data `Y` to the `self.data`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_features]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        knn : `int` or `None`, optional (default: `None`)
            If `None`, defaults to `self.knn`

        bandwidth : `float`, `callable`, or `None`, optional (default: `None`)
            If `None`, defaults to `self.bandwidth`

        bandwidth_scale : `float`, optional (default : `None`)
            Rescaling factor for bandwidth.
            If `None`, defaults to self.bandwidth_scale

        Returns
        -------

        K_yx: array-like, [n_samples_y, n_samples]
            kernel matrix where each row represents affinities of a single
            sample in `Y` to all samples in `self.data`.

        Raises
        ------

        ValueError: if the supplied data is the wrong shape
        """
        if knn is None:
            knn = self.knn
        if bandwidth is None:
            bandwidth = self.bandwidth
        if bandwidth_scale is None:
            bandwidth_scale = self.bandwidth_scale
        if knn > self.data.shape[0]:
            warnings.warn('Cannot set knn ({k}) to be greater than n_samples ({n}). Setting knn={n}'.format(k=knn, n=self.data_nu.shape[0]))
            knn = self.data_nu.shape[0]
        if knn_max is None:
            knn_max = self.data_nu.shape[0]
        Y = self._check_extension_shape(Y)
        if self.decay is None or self.thresh == 1:
            with _logger.log_task('KNN search'):
                K = self.knn_tree.kneighbors_graph(Y, n_neighbors=knn, mode='connectivity')
        else:
            with _logger.log_task('KNN search'):
                knn_tree = self.knn_tree
                search_knn = min(knn * self.search_multiplier, knn_max)
                distances, indices = knn_tree.kneighbors(Y, n_neighbors=search_knn)
                self._check_duplicates(distances, indices)
            with _logger.log_task('affinities'):
                if bandwidth is None:
                    bandwidth = distances[:, knn - 1]
                bandwidth = bandwidth * bandwidth_scale
                bandwidth = np.maximum(bandwidth, np.finfo(float).eps)
                radius = bandwidth * np.power(-1 * np.log(self.thresh), 1 / self.decay)
                update_idx = np.argwhere(np.max(distances, axis=1) < radius).reshape(-1)
                _logger.log_debug('search_knn = {}; {} remaining'.format(search_knn, len(update_idx)))
                if len(update_idx) > 0:
                    distances = [d for d in distances]
                    indices = [i for i in indices]
                search_knn = min(search_knn * self.search_multiplier, knn_max)
                while len(update_idx) > Y.shape[0] // 10 and search_knn < self.data_nu.shape[0] / 2 and (search_knn < knn_max):
                    dist_new, ind_new = knn_tree.kneighbors(Y[update_idx], n_neighbors=search_knn)
                    for i, idx in enumerate(update_idx):
                        distances[idx] = dist_new[i]
                        indices[idx] = ind_new[i]
                    update_idx = [i for i, d in enumerate(distances) if np.max(d) < (radius if isinstance(bandwidth, numbers.Number) else radius[i])]
                    _logger.log_debug('search_knn = {}; {} remaining'.format(search_knn, len(update_idx)))
                    search_knn = min(search_knn * self.search_multiplier, knn_max)
                if search_knn > self.data_nu.shape[0] / 2:
                    knn_tree = NearestNeighbors(n_neighbors=search_knn, algorithm='brute', n_jobs=self.n_jobs).fit(self.data_nu)
                if len(update_idx) > 0:
                    if search_knn == knn_max:
                        _logger.log_debug('knn search to knn_max ({}) on {}'.format(knn_max, len(update_idx)))
                        dist_new, ind_new = knn_tree.kneighbors(Y[update_idx], n_neighbors=search_knn)
                        for i, idx in enumerate(update_idx):
                            distances[idx] = dist_new[i]
                            indices[idx] = ind_new[i]
                    else:
                        _logger.log_debug('radius search on {}'.format(len(update_idx)))
                        dist_new, ind_new = knn_tree.radius_neighbors(Y[update_idx, :], radius=radius if isinstance(bandwidth, numbers.Number) else np.max(radius[update_idx]))
                        for i, idx in enumerate(update_idx):
                            distances[idx] = dist_new[i]
                            indices[idx] = ind_new[i]
                if isinstance(bandwidth, numbers.Number):
                    data = np.concatenate(distances) / bandwidth
                else:
                    data = np.concatenate([distances[i] / bandwidth[i] for i in range(len(distances))])
                indices = np.concatenate(indices)
                indptr = np.concatenate([[0], np.cumsum([len(d) for d in distances])])
                K = sparse.csr_matrix((data, indices, indptr), shape=(Y.shape[0], self.data_nu.shape[0]))
                K.data = np.exp(-1 * np.power(K.data, self.decay))
                K.data = np.where(np.isnan(K.data), 1, K.data)
                K.data[K.data < self.thresh] = 0
                K = K.tocoo()
                K.eliminate_zeros()
                K = K.tocsr()
        return K