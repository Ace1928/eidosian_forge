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
def _get_leaderboard_df(self, run, eval_metrics):
    import pandas as pd
    mlflow_client = MlflowClient()
    exp_id = _get_experiment_id()
    primary_metric_greater_is_better = self.evaluation_metrics[self.primary_metric].greater_is_better
    primary_metric_order = 'DESC' if primary_metric_greater_is_better else 'ASC'
    search_max_results = 100
    search_result = mlflow_client.search_runs(experiment_ids=exp_id, run_view_type=ViewType.ACTIVE_ONLY, max_results=search_max_results, order_by=[f'metrics.{self.primary_metric} {primary_metric_order}'])
    metric_names = self.evaluation_metrics.keys()
    leaderboard_items = []
    for old_run in search_result:
        if all((metric_key in old_run.data.metrics for metric_key in metric_names)):
            leaderboard_items.append({'Run ID': old_run.info.run_id, 'Run Time': datetime.datetime.fromtimestamp(old_run.info.start_time // 1000), **{metric_name: old_run.data.metrics[metric_key] for metric_name, metric_key in zip(metric_names, metric_names)}})
    top_leaderboard_items = [{'Model Rank': i + 1, **t} for i, t in enumerate(leaderboard_items[:2])]
    if len(top_leaderboard_items) == 2 and top_leaderboard_items[0][self.primary_metric] == top_leaderboard_items[1][self.primary_metric]:
        top_leaderboard_items[1]['Model Rank'] = '1'
    top_leaderboard_item_index_values = ['Best', '2nd Best'][:len(top_leaderboard_items)]
    latest_model_item = {'Run ID': run.info.run_id, 'Run Time': datetime.datetime.fromtimestamp(run.info.start_time // 1000), **eval_metrics['validation']}
    for i, leaderboard_item in enumerate(leaderboard_items):
        latest_value = latest_model_item[self.primary_metric]
        historical_value = leaderboard_item[self.primary_metric]
        if primary_metric_greater_is_better and latest_value >= historical_value or (not primary_metric_greater_is_better and latest_value <= historical_value):
            latest_model_item['Model Rank'] = str(i + 1)
            break
    else:
        latest_model_item['Model Rank'] = f'> {len(leaderboard_items)}'

    def sorter(m):
        if m == self.primary_metric:
            return (0, m)
        elif self.evaluation_metrics[m].custom_function is not None:
            return (1, m)
        else:
            return (2, m)
    metric_columns = sorted(metric_names, key=sorter)
    return pd.DataFrame.from_records([latest_model_item, *top_leaderboard_items], columns=['Model Rank', *metric_columns, 'Run Time', 'Run ID']).apply(lambda s: s.map(lambda x: f'{x:.6g}') if s.name in metric_names else s, axis=0).set_axis(['Latest'] + top_leaderboard_item_index_values, axis='index').transpose()