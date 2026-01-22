from functools import partial
import numpy as np
from sklearn.base import (
from sklearn.metrics._scorer import _Scorer, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.multiclass import _check_partial_fit_first_call
class ConsumingScorer(_Scorer):

    def __init__(self, registry=None):
        super().__init__(score_func=mean_squared_error, sign=1, kwargs={}, response_method='predict')
        self.registry = registry

    def _score(self, method_caller, clf, X, y, **kwargs):
        if self.registry is not None:
            self.registry.append(self)
        record_metadata_not_default(self, 'score', **kwargs)
        sample_weight = kwargs.get('sample_weight', None)
        return super()._score(method_caller, clf, X, y, sample_weight=sample_weight)