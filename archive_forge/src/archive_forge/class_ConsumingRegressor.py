from functools import partial
import numpy as np
from sklearn.base import (
from sklearn.metrics._scorer import _Scorer, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.multiclass import _check_partial_fit_first_call
class ConsumingRegressor(RegressorMixin, BaseEstimator):
    """A regressor consuming metadata.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.
    """

    def __init__(self, registry=None):
        self.registry = registry

    def partial_fit(self, X, y, sample_weight='default', metadata='default'):
        if self.registry is not None:
            self.registry.append(self)
        record_metadata_not_default(self, 'partial_fit', sample_weight=sample_weight, metadata=metadata)
        return self

    def fit(self, X, y, sample_weight='default', metadata='default'):
        if self.registry is not None:
            self.registry.append(self)
        record_metadata_not_default(self, 'fit', sample_weight=sample_weight, metadata=metadata)
        return self

    def predict(self, X, sample_weight='default', metadata='default'):
        pass