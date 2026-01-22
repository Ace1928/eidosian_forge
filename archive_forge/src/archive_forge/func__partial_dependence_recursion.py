from collections.abc import Iterable
import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles
from ..base import is_classifier, is_regressor
from ..ensemble import RandomForestRegressor
from ..ensemble._gb import BaseGradientBoosting
from ..ensemble._hist_gradient_boosting.gradient_boosting import (
from ..exceptions import NotFittedError
from ..tree import DecisionTreeRegressor
from ..utils import (
from ..utils._param_validation import (
from ..utils.extmath import cartesian
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._pd_utils import _check_feature_names, _get_feature_index
def _partial_dependence_recursion(est, grid, features):
    """Calculate partial dependence via the recursion method.

    The recursion method is in particular enabled for tree-based estimators.

    For each `grid` value, a weighted tree traversal is performed: if a split node
    involves an input feature of interest, the corresponding left or right branch
    is followed; otherwise both branches are followed, each branch being weighted
    by the fraction of training samples that entered that branch. Finally, the
    partial dependence is given by a weighted average of all the visited leaves
    values.

    This method is more efficient in terms of speed than the `'brute'` method
    (:func:`~sklearn.inspection._partial_dependence._partial_dependence_brute`).
    However, here, the partial dependence computation is done explicitly with the
    `X` used during training of `est`.

    Parameters
    ----------
    est : BaseEstimator
        A fitted estimator object implementing :term:`predict` or
        :term:`decision_function`. Multioutput-multiclass classifiers are not
        supported. Note that `'recursion'` is only supported for some tree-based
        estimators (namely
        :class:`~sklearn.ensemble.GradientBoostingClassifier`,
        :class:`~sklearn.ensemble.GradientBoostingRegressor`,
        :class:`~sklearn.ensemble.HistGradientBoostingClassifier`,
        :class:`~sklearn.ensemble.HistGradientBoostingRegressor`,
        :class:`~sklearn.tree.DecisionTreeRegressor`,
        :class:`~sklearn.ensemble.RandomForestRegressor`,
        ).

    grid : array-like of shape (n_points, n_target_features)
        The grid of feature values for which the partial dependence is calculated.
        Note that `n_points` is the number of points in the grid and `n_target_features`
        is the number of features you are doing partial dependence at.

    features : array-like of {int, str}
        The feature (e.g. `[0]`) or pair of interacting features
        (e.g. `[(0, 1)]`) for which the partial dependency should be computed.

    Returns
    -------
    averaged_predictions : array-like of shape (n_targets, n_points)
        The averaged predictions for the given `grid` of features values.
        Note that `n_targets` is the number of targets (e.g. 1 for binary
        classification, `n_tasks` for multi-output regression, and `n_classes` for
        multiclass classification) and `n_points` is the number of points in the `grid`.
    """
    averaged_predictions = est._compute_partial_dependence_recursion(grid, features)
    if averaged_predictions.ndim == 1:
        averaged_predictions = averaged_predictions.reshape(1, -1)
    return averaged_predictions