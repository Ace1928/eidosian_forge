import re
import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import (
from sklearn.linear_model import LinearRegression
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import metadata_routing
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.validation import check_is_fitted
def test_simple_metadata_routing():
    clf = WeightedMetaClassifier(estimator=NonConsumingClassifier())
    clf.fit(X, y)
    clf = WeightedMetaClassifier(estimator=NonConsumingClassifier())
    clf.fit(X, y, sample_weight=my_weights)
    clf = WeightedMetaClassifier(estimator=ConsumingClassifier())
    err_message = '[sample_weight] are passed but are not explicitly set as requested or not for ConsumingClassifier.fit'
    with pytest.raises(ValueError, match=re.escape(err_message)):
        clf.fit(X, y, sample_weight=my_weights)
    clf = WeightedMetaClassifier(estimator=ConsumingClassifier().set_fit_request(sample_weight=False))
    clf.fit(X, y, sample_weight=my_weights)
    check_recorded_metadata(clf.estimator_, 'fit')
    clf = WeightedMetaClassifier(estimator=ConsumingClassifier().set_fit_request(sample_weight=True))
    clf.fit(X, y, sample_weight=my_weights)
    check_recorded_metadata(clf.estimator_, 'fit', sample_weight=my_weights)
    clf = WeightedMetaClassifier(estimator=ConsumingClassifier().set_fit_request(sample_weight='alternative_weight'))
    clf.fit(X, y, alternative_weight=my_weights)
    check_recorded_metadata(clf.estimator_, 'fit', sample_weight=my_weights)