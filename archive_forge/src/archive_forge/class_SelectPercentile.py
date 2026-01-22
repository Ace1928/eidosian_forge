import warnings
from numbers import Integral, Real
import numpy as np
from scipy import special, stats
from scipy.sparse import issparse
from ..base import BaseEstimator, _fit_context
from ..preprocessing import LabelBinarizer
from ..utils import as_float_array, check_array, check_X_y, safe_mask, safe_sqr
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin
class SelectPercentile(_BaseFilter):
    """Select features according to a percentile of the highest scores.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.

        .. versionadded:: 0.18

    percentile : int, default=10
        Percent of features to keep.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores, None if `score_func` returned only scores.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif : Mutual information for a discrete target.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    mutual_info_regression : Mutual information for a continuous target.
    SelectKBest : Select features based on the k highest scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.

    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.

    This filter supports unsupervised feature selection that only requests `X` for
    computing the scores.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.feature_selection import SelectPercentile, chi2
    >>> X, y = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
    >>> X_new.shape
    (1797, 7)
    """
    _parameter_constraints: dict = {**_BaseFilter._parameter_constraints, 'percentile': [Interval(Real, 0, 100, closed='both')]}

    def __init__(self, score_func=f_classif, *, percentile=10):
        super().__init__(score_func=score_func)
        self.percentile = percentile

    def _get_support_mask(self):
        check_is_fitted(self)
        if self.percentile == 100:
            return np.ones(len(self.scores_), dtype=bool)
        elif self.percentile == 0:
            return np.zeros(len(self.scores_), dtype=bool)
        scores = _clean_nans(self.scores_)
        threshold = np.percentile(scores, 100 - self.percentile)
        mask = scores > threshold
        ties = np.where(scores == threshold)[0]
        if len(ties):
            max_feats = int(len(scores) * self.percentile / 100)
            kept_ties = ties[:max_feats - mask.sum()]
            mask[kept_ties] = True
        return mask

    def _more_tags(self):
        return {'requires_y': False}