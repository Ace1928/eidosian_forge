import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from ..base import BaseEstimator, ClassifierMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import rbf_kernel
from ..neighbors import NearestNeighbors
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.fixes import laplacian as csgraph_laplacian
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted
class LabelSpreading(BaseLabelPropagation):
    """LabelSpreading model for semi-supervised learning.

    This model is similar to the basic Label Propagation algorithm,
    but uses affinity matrix based on the normalized graph Laplacian
    and soft clamping across the labels.

    Read more in the :ref:`User Guide <label_propagation>`.

    Parameters
    ----------
    kernel : {'knn', 'rbf'} or callable, default='rbf'
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape (n_samples, n_features),
        and return a (n_samples, n_samples) shaped weight matrix.

    gamma : float, default=20
      Parameter for rbf kernel.

    n_neighbors : int, default=7
      Parameter for knn kernel which is a strictly positive integer.

    alpha : float, default=0.2
      Clamping factor. A value in (0, 1) that specifies the relative amount
      that an instance should adopt the information from its neighbors as
      opposed to its initial label.
      alpha=0 means keeping the initial label information; alpha=1 means
      replacing all initial information.

    max_iter : int, default=30
      Maximum number of iterations allowed.

    tol : float, default=1e-3
      Convergence tolerance: threshold to consider the system at steady
      state.

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Input array.

    classes_ : ndarray of shape (n_classes,)
        The distinct labels used in classifying instances.

    label_distributions_ : ndarray of shape (n_samples, n_classes)
        Categorical distribution for each item.

    transduction_ : ndarray of shape (n_samples,)
        Label assigned to each item during :term:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    LabelPropagation : Unregularized graph based semi-supervised learning.

    References
    ----------
    `Dengyong Zhou, Olivier Bousquet, Thomas Navin Lal, Jason Weston,
    Bernhard Schoelkopf. Learning with local and global consistency (2004)
    <https://citeseerx.ist.psu.edu/doc_view/pid/d74c37aabf2d5cae663007cbd8718175466aea8c>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import LabelSpreading
    >>> label_prop_model = LabelSpreading()
    >>> iris = datasets.load_iris()
    >>> rng = np.random.RandomState(42)
    >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> label_prop_model.fit(iris.data, labels)
    LabelSpreading(...)
    """
    _variant = 'spreading'
    _parameter_constraints: dict = {**BaseLabelPropagation._parameter_constraints}
    _parameter_constraints['alpha'] = [Interval(Real, 0, 1, closed='neither')]

    def __init__(self, kernel='rbf', *, gamma=20, n_neighbors=7, alpha=0.2, max_iter=30, tol=0.001, n_jobs=None):
        super().__init__(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter, tol=tol, n_jobs=n_jobs)

    def _build_graph(self):
        """Graph matrix for Label Spreading computes the graph laplacian"""
        if self.kernel == 'knn':
            self.nn_fit = None
        n_samples = self.X_.shape[0]
        affinity_matrix = self._get_kernel(self.X_)
        laplacian = csgraph_laplacian(affinity_matrix, normed=True)
        laplacian = -laplacian
        if sparse.issparse(laplacian):
            diag_mask = laplacian.row == laplacian.col
            laplacian.data[diag_mask] = 0.0
        else:
            laplacian.flat[::n_samples + 1] = 0.0
        return laplacian