from functools import partial
import numpy as np
from sklearn.base import (
from sklearn.metrics._scorer import _Scorer, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.multiclass import _check_partial_fit_first_call
class WeightedMetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """A meta-regressor which is also a consumer."""

    def __init__(self, estimator, registry=None):
        self.estimator = estimator
        self.registry = registry

    def fit(self, X, y, sample_weight=None, **fit_params):
        if self.registry is not None:
            self.registry.append(self)
        record_metadata(self, 'fit', sample_weight=sample_weight)
        params = process_routing(self, 'fit', sample_weight=sample_weight, **fit_params)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)
        return self

    def predict(self, X, **predict_params):
        params = process_routing(self, 'predict', **predict_params)
        return self.estimator_.predict(X, **params.estimator.predict)

    def get_metadata_routing(self):
        router = MetadataRouter(owner=self.__class__.__name__).add_self_request(self).add(estimator=self.estimator, method_mapping='one-to-one')
        return router