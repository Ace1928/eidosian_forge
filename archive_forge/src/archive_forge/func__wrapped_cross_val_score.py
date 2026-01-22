import numpy as np
from deap import tools, gp
from inspect import isclass
from .operator_utils import set_sample_weight
from sklearn.utils import indexable
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from collections import defaultdict
import warnings
from stopit import threading_timeoutable, TimeoutException
@threading_timeoutable(default='Timeout')
def _wrapped_cross_val_score(sklearn_pipeline, features, target, cv, scoring_function, sample_weight=None, groups=None, use_dask=False):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    sklearn_pipeline : pipeline object implementing 'fit'
        The object to use to fit the data.
    features : array-like of shape at least 2D
        The data to fit.
    target : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    cv: cross-validation generator
        Object to be used as a cross-validation generator.
    scoring_function : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    sample_weight : array-like, optional
        List of sample weights to balance (or un-balanace) the dataset target as needed
    groups: array-like {n_samples, }, optional
        Group labels for the samples used while splitting the dataset into train/test set
    use_dask : bool, default False
        Whether to use dask
    """
    sample_weight_dict = set_sample_weight(sklearn_pipeline.steps, sample_weight)
    features, target, groups = indexable(features, target, groups)
    cv_iter = list(cv.split(features, target, groups))
    scorer = check_scoring(sklearn_pipeline, scoring=scoring_function)
    if use_dask:
        try:
            import dask_ml.model_selection
            import dask
            from dask.delayed import Delayed
        except Exception as e:
            msg = "'use_dask' requires the optional dask and dask-ml depedencies.\n{}".format(e)
            raise ImportError(msg)
        dsk, keys, n_splits = dask_ml.model_selection._search.build_graph(estimator=sklearn_pipeline, cv=cv, scorer=scorer, candidate_params=[{}], X=features, y=target, groups=groups, fit_params=sample_weight_dict, refit=False, error_score=float('-inf'))
        cv_results = Delayed(keys[0], dsk)
        scores = [cv_results['split{}_test_score'.format(i)] for i in range(n_splits)]
        CV_score = dask.delayed(np.array)(scores)[:, 0]
        return dask.delayed(np.nanmean)(CV_score)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scores = [_fit_and_score(estimator=clone(sklearn_pipeline), X=features, y=target, scorer=scorer, train=train, test=test, verbose=0, parameters=None, error_score='raise', fit_params=sample_weight_dict, score_params=None) for train, test in cv_iter]
                if isinstance(scores[0], list):
                    CV_score = np.array(scores)[:, 0]
                elif isinstance(scores[0], dict):
                    from sklearn.model_selection._validation import _aggregate_score_dicts
                    CV_score = _aggregate_score_dicts(scores)['test_scores']
                else:
                    raise ValueError('Incorrect output format from _fit_and_score!')
                CV_score_mean = np.nanmean(CV_score)
            return CV_score_mean
        except TimeoutException:
            return 'Timeout'
        except Exception as e:
            return -float('inf')