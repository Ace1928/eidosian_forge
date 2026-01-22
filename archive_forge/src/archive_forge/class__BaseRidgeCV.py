import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy import linalg, optimize, sparse
from scipy.sparse import linalg as sp_linalg
from ..base import MultiOutputMixin, RegressorMixin, _fit_context, is_classifier
from ..exceptions import ConvergenceWarning
from ..metrics import check_scoring, get_scorer_names
from ..model_selection import GridSearchCV
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import _sparse_linalg_cg
from ..utils.metadata_routing import (
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, LinearModel, _preprocess_data, _rescale_data
from ._sag import sag_solver
class _BaseRidgeCV(LinearModel):
    _parameter_constraints: dict = {'alphas': ['array-like', Interval(Real, 0, None, closed='neither')], 'fit_intercept': ['boolean'], 'scoring': [StrOptions(set(get_scorer_names())), callable, None], 'cv': ['cv_object'], 'gcv_mode': [StrOptions({'auto', 'svd', 'eigen'}), None], 'store_cv_values': ['boolean'], 'alpha_per_target': ['boolean']}

    def __init__(self, alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None, cv=None, gcv_mode=None, store_cv_values=False, alpha_per_target=False):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.cv = cv
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
        self.alpha_per_target = alpha_per_target

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model with cv.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data. If using GCV, will be cast to float64
            if necessary.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        When sample_weight is provided, the selected hyperparameter may depend
        on whether we use leave-one-out cross-validation (cv=None or cv='auto')
        or another form of cross-validation, because only leave-one-out
        cross-validation takes the sample weights into account when computing
        the validation score.
        """
        cv = self.cv
        check_scalar_alpha = partial(check_scalar, target_type=numbers.Real, min_val=0.0, include_boundaries='neither')
        if isinstance(self.alphas, (np.ndarray, list, tuple)):
            n_alphas = 1 if np.ndim(self.alphas) == 0 else len(self.alphas)
            if n_alphas != 1:
                for index, alpha in enumerate(self.alphas):
                    alpha = check_scalar_alpha(alpha, f'alphas[{index}]')
            else:
                self.alphas[0] = check_scalar_alpha(self.alphas[0], 'alphas')
        alphas = np.asarray(self.alphas)
        if cv is None:
            estimator = _RidgeGCV(alphas, fit_intercept=self.fit_intercept, scoring=self.scoring, gcv_mode=self.gcv_mode, store_cv_values=self.store_cv_values, is_clf=is_classifier(self), alpha_per_target=self.alpha_per_target)
            estimator.fit(X, y, sample_weight=sample_weight)
            self.alpha_ = estimator.alpha_
            self.best_score_ = estimator.best_score_
            if self.store_cv_values:
                self.cv_values_ = estimator.cv_values_
        else:
            if self.store_cv_values:
                raise ValueError('cv!=None and store_cv_values=True are incompatible')
            if self.alpha_per_target:
                raise ValueError('cv!=None and alpha_per_target=True are incompatible')
            parameters = {'alpha': alphas}
            solver = 'sparse_cg' if sparse.issparse(X) else 'auto'
            model = RidgeClassifier if is_classifier(self) else Ridge
            gs = GridSearchCV(model(fit_intercept=self.fit_intercept, solver=solver), parameters, cv=cv, scoring=self.scoring)
            gs.fit(X, y, sample_weight=sample_weight)
            estimator = gs.best_estimator_
            self.alpha_ = gs.best_estimator_.alpha
            self.best_score_ = gs.best_score_
        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_
        self.n_features_in_ = estimator.n_features_in_
        if hasattr(estimator, 'feature_names_in_'):
            self.feature_names_in_ = estimator.feature_names_in_
        return self