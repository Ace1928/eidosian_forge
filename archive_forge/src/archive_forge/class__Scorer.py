import copy
import warnings
from collections import Counter
from functools import partial
from inspect import signature
from traceback import format_exc
from ..base import is_regressor
from ..utils import Bunch
from ..utils._param_validation import HasMethods, Hidden, StrOptions, validate_params
from ..utils._response import _get_response_values
from ..utils.metadata_routing import (
from ..utils.validation import _check_response_method
from . import (
from .cluster import (
class _Scorer(_BaseScorer):

    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        """Evaluate the response method of `estimator` on `X` and `y_true`.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring.

        X : {array-like, sparse matrix}
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        self._warn_overlap(message='There is an overlap between set kwargs of this scorer instance and passed metadata. Please pass them either as kwargs to `make_scorer` or metadata, but not both.', kwargs=kwargs)
        pos_label = None if is_regressor(estimator) else self._get_pos_label()
        response_method = _check_response_method(estimator, self._response_method)
        y_pred = method_caller(estimator, response_method.__name__, X, pos_label=pos_label)
        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)