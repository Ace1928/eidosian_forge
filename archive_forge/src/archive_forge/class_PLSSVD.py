import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.linalg import svd
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_consistent_length
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import svd_flip
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import FLOAT_DTYPES, check_is_fitted
class PLSSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Partial Least Square SVD.

    This transformer simply performs a SVD on the cross-covariance matrix
    `X'Y`. It is able to project both the training data `X` and the targets
    `Y`. The training data `X` is projected on the left singular vectors, while
    the targets are projected on the right singular vectors.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        The number of components to keep. Should be in `[1,
        min(n_samples, n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If `False`, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    y_weights_ : ndarray of (n_targets, n_components)
        The right singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.
    CCA : Canonical Correlation Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cross_decomposition import PLSSVD
    >>> X = np.array([[0., 0., 1.],
    ...               [1., 0., 0.],
    ...               [2., 2., 2.],
    ...               [2., 5., 4.]])
    >>> Y = np.array([[0.1, -0.2],
    ...               [0.9, 1.1],
    ...               [6.2, 5.9],
    ...               [11.9, 12.3]])
    >>> pls = PLSSVD(n_components=2).fit(X, Y)
    >>> X_c, Y_c = pls.transform(X, Y)
    >>> X_c.shape, Y_c.shape
    ((4, 2), (4, 2))
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left')], 'scale': ['boolean'], 'copy': ['boolean']}

    def __init__(self, n_components=2, *, scale=True, copy=True):
        self.n_components = n_components
        self.scale = scale
        self.copy = copy

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Targets.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        check_consistent_length(X, Y)
        X = self._validate_data(X, dtype=np.float64, copy=self.copy, ensure_min_samples=2)
        Y = check_array(Y, input_name='Y', dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n_components = self.n_components
        rank_upper_bound = min(X.shape[0], X.shape[1], Y.shape[1])
        if n_components > rank_upper_bound:
            raise ValueError(f'`n_components` upper bound is {rank_upper_bound}. Got {n_components} instead. Reduce `n_components`.')
        X, Y, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(X, Y, self.scale)
        C = np.dot(X.T, Y)
        U, s, Vt = svd(C, full_matrices=False)
        U = U[:, :n_components]
        Vt = Vt[:n_components]
        U, Vt = svd_flip(U, Vt)
        V = Vt.T
        self.x_weights_ = U
        self.y_weights_ = V
        self._n_features_out = self.x_weights_.shape[1]
        return self

    def transform(self, X, Y=None):
        """
        Apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to be transformed.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets),                 default=None
            Targets.

        Returns
        -------
        x_scores : array-like or tuple of array-like
            The transformed data `X_transformed` if `Y is not None`,
            `(X_transformed, Y_transformed)` otherwise.
        """
        check_is_fitted(self)
        X = self._validate_data(X, dtype=np.float64, reset=False)
        Xr = (X - self._x_mean) / self._x_std
        x_scores = np.dot(Xr, self.x_weights_)
        if Y is not None:
            Y = check_array(Y, input_name='Y', ensure_2d=False, dtype=np.float64)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Yr = (Y - self._y_mean) / self._y_std
            y_scores = np.dot(Yr, self.y_weights_)
            return (x_scores, y_scores)
        return x_scores

    def fit_transform(self, X, y=None):
        """Learn and apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        y : array-like of shape (n_samples,) or (n_samples, n_targets),                 default=None
            Targets.

        Returns
        -------
        out : array-like or tuple of array-like
            The transformed data `X_transformed` if `Y is not None`,
            `(X_transformed, Y_transformed)` otherwise.
        """
        return self.fit(X, y).transform(X, y)