import importlib
import logging
import os
import sys
import time
import cloudpickle
from packaging.version import Version
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.recipes.artifacts import DataframeArtifact, TransformerArtifact
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.step import get_pandas_data_profiles, validate_classification_config
from mlflow.recipes.utils.tracking import TrackingConfig, get_recipe_tracking_config
def _get_output_feature_names(transformer, num_features, input_features):
    import sklearn
    if Version(sklearn.__version__) < Version('1.0.0'):
        return _generate_feature_names(num_features)
    try:
        return transformer.get_feature_names_out(input_features)
    except Exception as e:
        _logger.warning(f'Failed to get output feature names with `get_feature_names_out`: {e}. Falling back to using auto-generated feature names.')
        return _generate_feature_names(num_features)