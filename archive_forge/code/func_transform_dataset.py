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
def transform_dataset(dataset):
    features = dataset.drop(columns=[self.target_col])
    transformed_features = transformer.transform(features)
    if not isinstance(transformed_features, pd.DataFrame):
        num_features = transformed_features.shape[1]
        columns = _get_output_feature_names(transformer, num_features, features.columns)
        transformed_features = pd.DataFrame(transformed_features, columns=columns)
    transformed_features[self.target_col] = dataset[self.target_col].values
    return transformed_features