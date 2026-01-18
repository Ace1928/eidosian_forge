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
@pytest.mark.parametrize('estimator', UNSUPPORTED_ESTIMATORS)
def test_unsupported_estimators_get_metadata_routing(estimator):
    """Test that get_metadata_routing is not implemented on meta-estimators for
    which we haven't implemented routing yet."""
    with pytest.raises(NotImplementedError):
        estimator.get_metadata_routing()