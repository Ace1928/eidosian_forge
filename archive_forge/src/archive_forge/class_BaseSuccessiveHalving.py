from abc import abstractmethod
from copy import deepcopy
from math import ceil, floor, log
from numbers import Integral, Real
import numpy as np
from ..base import _fit_context, is_classifier
from ..metrics._scorer import get_scorer_names
from ..utils import resample
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import _num_samples
from . import ParameterGrid, ParameterSampler
from ._search import BaseSearchCV
from ._split import _yields_constant_splits, check_cv
class BaseSuccessiveHalving(BaseSearchCV):
    """Implements successive halving.

    Ref:
    Almost optimal exploration in multi-armed bandits, ICML 13
    Zohar Karnin, Tomer Koren, Oren Somekh
    """
    _parameter_constraints: dict = {**BaseSearchCV._parameter_constraints, 'scoring': [StrOptions(set(get_scorer_names())), callable, None], 'random_state': ['random_state'], 'max_resources': [Interval(Integral, 0, None, closed='neither'), StrOptions({'auto'})], 'min_resources': [Interval(Integral, 0, None, closed='neither'), StrOptions({'exhaust', 'smallest'})], 'resource': [str], 'factor': [Interval(Real, 0, None, closed='neither')], 'aggressive_elimination': ['boolean']}
    _parameter_constraints.pop('pre_dispatch')

    def __init__(self, estimator, *, scoring=None, n_jobs=None, refit=True, cv=5, verbose=0, random_state=None, error_score=np.nan, return_train_score=True, max_resources='auto', min_resources='exhaust', resource='n_samples', factor=3, aggressive_elimination=False):
        super().__init__(estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose, error_score=error_score, return_train_score=return_train_score)
        self.random_state = random_state
        self.max_resources = max_resources
        self.resource = resource
        self.factor = factor
        self.min_resources = min_resources
        self.aggressive_elimination = aggressive_elimination

    def _check_input_parameters(self, X, y, split_params):
        if not _yields_constant_splits(self._checked_cv_orig):
            raise ValueError('The cv parameter must yield consistent folds across calls to split(). Set its random_state to an int, or set shuffle=False.')
        if self.resource != 'n_samples' and self.resource not in self.estimator.get_params():
            raise ValueError(f'Cannot use resource={self.resource} which is not supported by estimator {self.estimator.__class__.__name__}')
        if isinstance(self, HalvingRandomSearchCV):
            if self.min_resources == self.n_candidates == 'exhaust':
                raise ValueError("n_candidates and min_resources cannot be both set to 'exhaust'.")
        self.min_resources_ = self.min_resources
        if self.min_resources_ in ('smallest', 'exhaust'):
            if self.resource == 'n_samples':
                n_splits = self._checked_cv_orig.get_n_splits(X, y, **split_params)
                magic_factor = 2
                self.min_resources_ = n_splits * magic_factor
                if is_classifier(self.estimator):
                    y = self._validate_data(X='no_validation', y=y)
                    check_classification_targets(y)
                    n_classes = np.unique(y).shape[0]
                    self.min_resources_ *= n_classes
            else:
                self.min_resources_ = 1
        self.max_resources_ = self.max_resources
        if self.max_resources_ == 'auto':
            if not self.resource == 'n_samples':
                raise ValueError("resource can only be 'n_samples' when max_resources='auto'")
            self.max_resources_ = _num_samples(X)
        if self.min_resources_ > self.max_resources_:
            raise ValueError(f'min_resources_={self.min_resources_} is greater than max_resources_={self.max_resources_}.')
        if self.min_resources_ == 0:
            raise ValueError(f'min_resources_={self.min_resources_}: you might have passed an empty dataset X.')

    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """Custom refit callable to return the index of the best candidate.

        We want the best candidate out of the last iteration. By default
        BaseSearchCV would return the best candidate out of all iterations.

        Currently, we only support for a single metric thus `refit` and
        `refit_metric` are not required.
        """
        last_iter = np.max(results['iter'])
        last_iter_indices = np.flatnonzero(results['iter'] == last_iter)
        test_scores = results['mean_test_score'][last_iter_indices]
        if np.isnan(test_scores).all():
            best_idx = 0
        else:
            best_idx = np.nanargmax(test_scores)
        return last_iter_indices[best_idx]

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_output), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        self._checked_cv_orig = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        routed_params = self._get_routed_params_for_fit(params)
        self._check_input_parameters(X=X, y=y, split_params=routed_params.splitter.split)
        self._n_samples_orig = _num_samples(X)
        super().fit(X, y=y, **params)
        self.best_score_ = self.cv_results_['mean_test_score'][self.best_index_]
        return self

    def _run_search(self, evaluate_candidates):
        candidate_params = self._generate_candidate_params()
        if self.resource != 'n_samples' and any((self.resource in candidate for candidate in candidate_params)):
            raise ValueError(f'Cannot use parameter {self.resource} as the resource since it is part of the searched parameters.')
        n_required_iterations = 1 + floor(log(len(candidate_params), self.factor))
        if self.min_resources == 'exhaust':
            last_iteration = n_required_iterations - 1
            self.min_resources_ = max(self.min_resources_, self.max_resources_ // self.factor ** last_iteration)
        n_possible_iterations = 1 + floor(log(self.max_resources_ // self.min_resources_, self.factor))
        if self.aggressive_elimination:
            n_iterations = n_required_iterations
        else:
            n_iterations = min(n_possible_iterations, n_required_iterations)
        if self.verbose:
            print(f'n_iterations: {n_iterations}')
            print(f'n_required_iterations: {n_required_iterations}')
            print(f'n_possible_iterations: {n_possible_iterations}')
            print(f'min_resources_: {self.min_resources_}')
            print(f'max_resources_: {self.max_resources_}')
            print(f'aggressive_elimination: {self.aggressive_elimination}')
            print(f'factor: {self.factor}')
        self.n_resources_ = []
        self.n_candidates_ = []
        for itr in range(n_iterations):
            power = itr
            if self.aggressive_elimination:
                power = max(0, itr - n_required_iterations + n_possible_iterations)
            n_resources = int(self.factor ** power * self.min_resources_)
            n_resources = min(n_resources, self.max_resources_)
            self.n_resources_.append(n_resources)
            n_candidates = len(candidate_params)
            self.n_candidates_.append(n_candidates)
            if self.verbose:
                print('-' * 10)
                print(f'iter: {itr}')
                print(f'n_candidates: {n_candidates}')
                print(f'n_resources: {n_resources}')
            if self.resource == 'n_samples':
                cv = _SubsampleMetaSplitter(base_cv=self._checked_cv_orig, fraction=n_resources / self._n_samples_orig, subsample_test=True, random_state=self.random_state)
            else:
                candidate_params = [c.copy() for c in candidate_params]
                for candidate in candidate_params:
                    candidate[self.resource] = n_resources
                cv = self._checked_cv_orig
            more_results = {'iter': [itr] * n_candidates, 'n_resources': [n_resources] * n_candidates}
            results = evaluate_candidates(candidate_params, cv, more_results=more_results)
            n_candidates_to_keep = ceil(n_candidates / self.factor)
            candidate_params = _top_k(results, n_candidates_to_keep, itr)
        self.n_remaining_candidates_ = len(candidate_params)
        self.n_required_iterations_ = n_required_iterations
        self.n_possible_iterations_ = n_possible_iterations
        self.n_iterations_ = n_iterations

    @abstractmethod
    def _generate_candidate_params(self):
        pass

    def _more_tags(self):
        tags = deepcopy(super()._more_tags())
        tags['_xfail_checks'].update({'check_fit2d_1sample': 'Fail during parameter check since min/max resources requires more samples'})
        return tags