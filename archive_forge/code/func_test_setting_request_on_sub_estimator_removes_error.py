import copy
import re
import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.covariance import GraphicalLassoCV
from sklearn.ensemble import (
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.experimental import (
from sklearn.feature_selection import (
from sklearn.impute import IterativeImputer
from sklearn.linear_model import (
from sklearn.model_selection import (
from sklearn.multiclass import (
from sklearn.multioutput import (
from sklearn.pipeline import FeatureUnion
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tests.metadata_routing_common import (
from sklearn.utils.metadata_routing import MetadataRouter
@pytest.mark.parametrize('metaestimator', METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_setting_request_on_sub_estimator_removes_error(metaestimator):
    if 'estimator' not in metaestimator:
        return

    def set_request(estimator, method_name):
        set_request_for_method = getattr(estimator, f'set_{method_name}_request')
        set_request_for_method(sample_weight=True, metadata=True)
        if is_classifier(estimator) and method_name == 'partial_fit':
            set_request_for_method(classes=True)
    cls = metaestimator['metaestimator']
    X = metaestimator['X']
    y = metaestimator['y']
    routing_methods = metaestimator['estimator_routing_methods']
    preserves_metadata = metaestimator.get('preserves_metadata', True)
    for method_name in routing_methods:
        for key in ['sample_weight', 'metadata']:
            val = {'sample_weight': sample_weight, 'metadata': metadata}[key]
            method_kwargs = {key: val}
            kwargs, (estimator, registry), (scorer, _), (cv, _) = get_init_args(metaestimator, sub_estimator_consumes=True)
            if scorer:
                set_request(scorer, 'score')
            if cv:
                cv.set_split_request(groups=True, metadata=True)
            set_request(estimator, method_name)
            instance = cls(**kwargs)
            method = getattr(instance, method_name)
            extra_method_args = metaestimator.get('method_args', {}).get(method_name, {})
            method(X, y, **method_kwargs, **extra_method_args)
            assert registry
            if preserves_metadata is True:
                for estimator in registry:
                    check_recorded_metadata(estimator, method_name, **method_kwargs)
            elif preserves_metadata == 'subset':
                for estimator in registry:
                    check_recorded_metadata(estimator, method_name, split_params=method_kwargs.keys(), **method_kwargs)