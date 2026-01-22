import warnings
from numbers import Integral, Real
import numpy as np
from ..base import _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, validate_params
from . import EmpiricalCovariance, empirical_covariance
class OAS(EmpiricalCovariance):
    """Oracle Approximating Shrinkage Estimator as proposed in [1]_.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    shrinkage_ : float
      coefficient in the convex combination used for the computation
      of the shrunk estimate. Range is [0, 1].

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    EllipticEnvelope : An object for detecting outliers in
        a Gaussian distributed dataset.
    EmpiricalCovariance : Maximum likelihood covariance estimator.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    GraphicalLassoCV : Sparse inverse covariance with cross-validated
        choice of the l1 penalty.
    LedoitWolf : LedoitWolf Estimator.
    MinCovDet : Minimum Covariance Determinant
        (robust estimator of covariance).
    ShrunkCovariance : Covariance estimator with shrinkage.

    Notes
    -----
    The regularised covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features),

    where mu = trace(cov) / n_features and shrinkage is given by the OAS formula
    (see [1]_).

    The shrinkage formulation implemented here differs from Eq. 23 in [1]_. In
    the original article, formula (23) states that 2/p (p being the number of
    features) is multiplied by Trace(cov*cov) in both the numerator and
    denominator, but this operation is omitted because for a large p, the value
    of 2/p is so small that it doesn't affect the value of the estimator.

    References
    ----------
    .. [1] :arxiv:`"Shrinkage algorithms for MMSE covariance estimation.",
           Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
           IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
           <0907.4698>`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import OAS
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> real_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0],
    ...                             cov=real_cov,
    ...                             size=500)
    >>> oas = OAS().fit(X)
    >>> oas.covariance_
    array([[0.7533..., 0.2763...],
           [0.2763..., 0.3964...]])
    >>> oas.precision_
    array([[ 1.7833..., -1.2431... ],
           [-1.2431...,  3.3889...]])
    >>> oas.shrinkage_
    0.0195...
    """

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the Oracle Approximating Shrinkage covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        covariance, shrinkage = _oas(X - self.location_, assume_centered=True)
        self.shrinkage_ = shrinkage
        self._set_covariance(covariance)
        return self