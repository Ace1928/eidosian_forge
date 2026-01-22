import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Tuple
import pandas as pd
import mlflow
from mlflow import MlflowException
from mlflow.models import EvaluationMetric
from mlflow.models.evaluation.default_evaluator import (
from mlflow.recipes.utils.metrics import RecipeMetric, _load_custom_metrics
def _create_model_automl(X, y, task: str, extended_task: str, step_config: Dict[str, Any], recipe_root: str, evaluation_metrics: Dict[str, RecipeMetric], primary_metric: str) -> Tuple['BaseEstimator', Dict[str, Any]]:
    try:
        from flaml import AutoML
    except ImportError:
        raise MlflowException('Please install FLAML to use AutoML!')
    try:
        if primary_metric in _MLFLOW_TO_FLAML_METRICS and primary_metric in evaluation_metrics:
            metric = _MLFLOW_TO_FLAML_METRICS[primary_metric]
            if primary_metric == 'roc_auc' and extended_task == 'classification/multiclass':
                metric = 'roc_auc_ovr'
        elif primary_metric in _SKLEARN_METRICS and primary_metric in evaluation_metrics:
            metric = _create_sklearn_metric_flaml(primary_metric, -1 if evaluation_metrics[primary_metric].greater_is_better else 1, 'macro' if extended_task in ['classification/multiclass'] else 'binary')
        elif primary_metric in evaluation_metrics:
            metric = _create_custom_metric_flaml(task, primary_metric, -1 if evaluation_metrics[primary_metric].greater_is_better else 1, _load_custom_metrics(recipe_root, [evaluation_metrics[primary_metric]])[0])
        else:
            raise MlflowException(f'There is no FLAML alternative or custom metric for {primary_metric} metric.')
        automl_settings = step_config.get('flaml_params', {})
        automl_settings['time_budget'] = step_config.get('time_budget_secs', _AUTOML_DEFAULT_TIME_BUDGET)
        automl_settings['metric'] = metric
        automl_settings['task'] = task
        mlflow.autolog(disable=True)
        automl = AutoML()
        automl.fit(X, y, **automl_settings)
        mlflow.autolog(disable=False, log_models=False)
        if automl.model is None:
            raise MlflowException('AutoML (FLAML) could not train a suitable algorithm. Maybe you should increase `time_budget_secs`parameter to give AutoML process more time.')
        return (automl.model.estimator, automl.best_config)
    except Exception as e:
        _logger.warning(e, exc_info=e, stack_info=True)
        raise MlflowException(f'Error has occurred during training of AutoML model using FLAML: {e!r}')