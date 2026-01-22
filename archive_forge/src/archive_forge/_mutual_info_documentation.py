from numbers import Integral
import numpy as np
from scipy.sparse import issparse
from scipy.special import digamma
from ..metrics.cluster import mutual_info_score
from ..neighbors import KDTree, NearestNeighbors
from ..preprocessing import scale
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_array, check_X_y
Estimate mutual information for a discrete target variable.

    Mutual information (MI) [1]_ between two random variables is a non-negative
    value, which measures the dependency between the variables. It is equal
    to zero if and only if two random variables are independent, and higher
    values mean higher dependency.

    The function relies on nonparametric methods based on entropy estimation
    from k-nearest neighbors distances as described in [2]_ and [3]_. Both
    methods are based on the idea originally proposed in [4]_.

    It can be used for univariate features selection, read more in the
    :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target vector.

    discrete_features : 'auto', bool or array-like, default='auto'
        If bool, then determines whether to consider all features discrete
        or continuous. If array, then it should be either a boolean mask
        with shape (n_features,) or array with indices of discrete features.
        If 'auto', it is assigned to False for dense `X` and to True for
        sparse `X`.

    n_neighbors : int, default=3
        Number of neighbors to use for MI estimation for continuous variables,
        see [2]_ and [3]_. Higher values reduce variance of the estimation, but
        could introduce a bias.

    copy : bool, default=True
        Whether to make a copy of the given data. If set to False, the initial
        data will be overwritten.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for adding small noise to
        continuous variables in order to remove repeated values.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    mi : ndarray, shape (n_features,)
        Estimated mutual information between each feature and the target in
        nat units.

    Notes
    -----
    1. The term "discrete features" is used instead of naming them
       "categorical", because it describes the essence more accurately.
       For example, pixel intensities of an image are discrete features
       (but hardly categorical) and you will get better results if mark them
       as such. Also note, that treating a continuous variable as discrete and
       vice versa will usually give incorrect results, so be attentive about
       that.
    2. True mutual information can't be negative. If its estimate turns out
       to be negative, it is replaced by zero.

    References
    ----------
    .. [1] `Mutual Information
           <https://en.wikipedia.org/wiki/Mutual_information>`_
           on Wikipedia.
    .. [2] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [3] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
    .. [4] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
           of a Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.feature_selection import mutual_info_classif
    >>> X, y = make_classification(
    ...     n_samples=100, n_features=10, n_informative=2, n_clusters_per_class=1,
    ...     shuffle=False, random_state=42
    ... )
    >>> mutual_info_classif(X, y)
    array([0.58..., 0.10..., 0.19..., 0.09... , 0.        ,
           0.     , 0.     , 0.     , 0.      , 0.        ])
    