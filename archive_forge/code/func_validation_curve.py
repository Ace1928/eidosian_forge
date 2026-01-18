import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from functools import partial
from numbers import Real
from traceback import format_exc
import numpy as np
import scipy.sparse as sp
from joblib import logger
from ..base import clone, is_classifier
from ..exceptions import FitFailedWarning, UnsetMetadataPassedError
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..preprocessing import LabelEncoder
from ..utils import Bunch, _safe_indexing, check_random_state, indexable
from ..utils._param_validation import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_method_params, _num_samples
from ._split import check_cv
@validate_params({'estimator': [HasMethods(['fit'])], 'X': ['array-like', 'sparse matrix'], 'y': ['array-like', None], 'param_name': [str], 'param_range': ['array-like'], 'groups': ['array-like', None], 'cv': ['cv_object'], 'scoring': [StrOptions(set(get_scorer_names())), callable, None], 'n_jobs': [Integral, None], 'pre_dispatch': [Integral, str], 'verbose': ['verbose'], 'error_score': [StrOptions({'raise'}), Real], 'fit_params': [dict, None]}, prefer_skip_nested_validation=False)
def validation_curve(estimator, X, y, *, param_name, param_range, groups=None, cv=None, scoring=None, n_jobs=None, pre_dispatch='all', verbose=0, error_score=np.nan, fit_params=None):
    """Validation curve.

    Determine training and test scores for varying parameter values.

    Compute scores for an estimator with different values of a specified
    parameter. This is similar to grid search with one parameter. However, this
    will also compute training scores and is merely a utility for plotting the
    results.

    Read more in the :ref:`User Guide <validation_curve>`.

    Parameters
    ----------
    estimator : object type that implements the "fit" method
        An object of that type which is cloned for each validation. It must
        also implement "predict" unless `scoring` is a callable that doesn't
        rely on "predict" to compute a score.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Target relative to X for classification or regression;
        None for unsupervised learning.

    param_name : str
        Name of the parameter that will be varied.

    param_range : array-like of shape (n_values,)
        The values of the parameter that will be evaluated.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the combinations of each parameter
        value and each cross-validation split.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch : int or str, default='all'
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The str can
        be an expression like '2*n_jobs'.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

        .. versionadded:: 0.24

    Returns
    -------
    train_scores : array of shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : array of shape (n_ticks, n_cv_folds)
        Scores on test set.

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import validation_curve
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=1_000, random_state=0)
    >>> logistic_regression = LogisticRegression()
    >>> param_name, param_range = "C", np.logspace(-8, 3, 10)
    >>> train_scores, test_scores = validation_curve(
    ...     logistic_regression, X, y, param_name=param_name, param_range=param_range
    ... )
    >>> print(f"The average train accuracy is {train_scores.mean():.2f}")
    The average train accuracy is 0.81
    >>> print(f"The average test accuracy is {test_scores.mean():.2f}")
    The average test accuracy is 0.81
    """
    X, y, groups = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch, verbose=verbose)
    results = parallel((delayed(_fit_and_score)(clone(estimator), X, y, scorer=scorer, train=train, test=test, verbose=verbose, parameters={param_name: v}, fit_params=fit_params, score_params=None, return_train_score=True, error_score=error_score) for train, test in cv.split(X, y, groups) for v in param_range))
    n_params = len(param_range)
    results = _aggregate_score_dicts(results)
    train_scores = results['train_scores'].reshape(-1, n_params).T
    test_scores = results['test_scores'].reshape(-1, n_params).T
    return (train_scores, test_scores)