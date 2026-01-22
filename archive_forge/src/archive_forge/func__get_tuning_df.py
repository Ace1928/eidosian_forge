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
def _get_tuning_df(self, run, params=None):
    exp_id = _get_experiment_id()
    primary_metric_tag = f'metrics.{self.primary_metric}'
    order_str = 'DESC' if self.evaluation_metrics_greater_is_better[self.primary_metric] else 'ASC'
    tuning_runs = mlflow.search_runs([exp_id], filter_string=f"tags.mlflow.parentRunId like '{run.info.run_id}'", order_by=[f'{primary_metric_tag} {order_str}', 'attribute.start_time ASC'])
    if params:
        params = [f'params.{param}' for param in params]
        tuning_runs = tuning_runs.filter([f'metrics.{self.primary_metric}', *params])
    else:
        tuning_runs = tuning_runs.filter([f'metrics.{self.primary_metric}'])
    tuning_runs = tuning_runs.reset_index().rename(columns={'index': 'Model Rank', primary_metric_tag: self.primary_metric})
    tuning_runs['Model Rank'] += 1
    return tuning_runs.head(10)