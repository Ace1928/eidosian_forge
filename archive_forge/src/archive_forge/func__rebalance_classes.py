import datetime
import importlib
import logging
import os
import re
import shutil
import sys
import warnings
import cloudpickle
import yaml
import mlflow
from mlflow.entities import SourceType, ViewType
from mlflow.environment_variables import MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME
from mlflow.exceptions import BAD_REQUEST, INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.recipes.artifacts import (
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.metrics import (
from mlflow.recipes.utils.step import (
from mlflow.recipes.utils.tracking import (
from mlflow.recipes.utils.wrapped_recipe_model import WrappedRecipeModel
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.databricks_utils import get_databricks_env_vars, get_databricks_run_url
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.string_utils import strip_prefix
def _rebalance_classes(self, train_df):
    import pandas as pd
    resampling_minority_percentage = self.step_config.get('resampling_minority_percentage', _REBALANCING_DEFAULT_RATIO)
    df_positive_class = train_df[train_df[self.target_col] == self.positive_class]
    df_negative_class = train_df[train_df[self.target_col] != self.positive_class]
    if len(df_positive_class) > len(df_negative_class):
        df_minority_class, df_majority_class = (df_negative_class, df_positive_class)
    else:
        df_minority_class, df_majority_class = (df_positive_class, df_negative_class)
    original_minority_percentage = len(df_minority_class) / len(train_df)
    if original_minority_percentage >= resampling_minority_percentage:
        _logger.info(f'Class imbalance of {original_minority_percentage:.2f} is better than {resampling_minority_percentage}, no need to rebalance')
        return train_df
    _logger.info(f'Detected class imbalance: minority class percentage is {original_minority_percentage:.2f}')
    majority_class_target = int(len(df_minority_class) * (1 - resampling_minority_percentage) / resampling_minority_percentage)
    df_majority_downsampled = df_majority_class.sample(majority_class_target)
    _logger.info(f'After downsampling: minority class percentage is {resampling_minority_percentage:.2f}')
    return pd.concat([df_minority_class, df_majority_downsampled], axis=0).sample(frac=1)