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
def _tune_and_get_best_estimator_params(self, parent_run_id, estimator_hardcoded_params, estimator_fn, X_train, y_train, validation_df):
    tuning_params = self.step_config['tuning']
    try:
        from hyperopt import SparkTrials, Trials, fmin, space_eval
    except ModuleNotFoundError:
        raise MlflowException('Hyperopt not installed and is required if tuning is enabled', error_code=BAD_REQUEST)

    def objective(X_train, y_train, validation_df, hyperparameter_args, on_worker=False):
        run_name = self.tracking_config.run_name
        if on_worker:
            client = MlflowClient()
            parent_tags = client.get_run(parent_run_id).data.tags
            child_run = client.create_run(_get_experiment_id(), tags={**parent_tags, 'mlflow.parentRunId': parent_run_id}, run_name=run_name)
            run_args = {'run_id': child_run.info.run_id}
        else:
            run_args = {'run_name': run_name, 'nested': True}
        with mlflow.start_run(**run_args) as tuning_run:
            estimator_args = dict(estimator_hardcoded_params, **hyperparameter_args)
            estimator = estimator_fn(estimator_args)
            sample_fraction = self.step_config['sample_fraction']
            if tuning_params['parallelism'] > 1:
                X_train = X_train.value
                y_train = y_train.value
                validation_df = validation_df.value
            X_train_sampled = X_train.sample(frac=sample_fraction, random_state=42)
            y_train_sampled = y_train.sample(frac=sample_fraction, random_state=42)
            fitted_estimator, _ = self._fitted_estimator(estimator, X_train_sampled, y_train_sampled)
            logged_estimator = self._log_estimator_to_mlflow(fitted_estimator, X_train_sampled, on_worker=on_worker)
            eval_result = mlflow.evaluate(model=logged_estimator.model_uri, data=validation_df, targets=self.target_col, model_type=_get_model_type_from_template(self.recipe), evaluators='default', extra_metrics=_load_custom_metrics(self.recipe_root, self.evaluation_metrics.values()), evaluator_config={'log_model_explainability': False, 'pos_label': self.positive_class})
            autologged_params = mlflow.get_run(run_id=tuning_run.info.run_id).data.params
            manual_log_params = {}
            for param_name, param_value in estimator_args.items():
                if param_name in autologged_params:
                    if not TrainStep.is_tuning_param_equal(param_value, autologged_params[param_name]):
                        _logger.warning(f'Failed to log search space parameter due to conflict. Attempted to log search space parameter {param_name} as {param_value}, but {param_name} is already logged as {autologged_params[param_name]} during training. Are you passing `estimator_params` properly to the  estimator?')
                else:
                    manual_log_params[param_name] = param_value
            if len(manual_log_params) > 0:
                mlflow.log_params(manual_log_params)
            transformed_metrics = transform_multiclass_metrics_dict(eval_result.metrics, self.extended_task)
            sign = -1 if self.evaluation_metrics_greater_is_better[self.primary_metric] else 1
            return sign * transformed_metrics[self.primary_metric]
    search_space = TrainStep.construct_search_space_from_yaml(tuning_params['parameters'])
    algo_type, algo_name = tuning_params['algorithm'].rsplit('.', 1)
    tuning_algo = getattr(importlib.import_module(algo_type, 'hyperopt'), algo_name)
    max_trials = tuning_params['max_trials']
    parallelism = tuning_params['parallelism']
    if parallelism > 1:
        from pyspark.sql import SparkSession
        spark_session = SparkSession.builder.config('spark.databricks.mlflow.trackHyperopt.enabled', 'false').getOrCreate()
        sc = spark_session.sparkContext
        X_train = sc.broadcast(X_train)
        y_train = sc.broadcast(y_train)
        validation_df = sc.broadcast(validation_df)
        hp_trials = SparkTrials(parallelism, spark_session=spark_session)
        on_worker = True
    else:
        hp_trials = Trials()
        on_worker = False
    fmin_kwargs = {'fn': lambda params: objective(X_train, y_train, validation_df, params, on_worker=on_worker), 'space': search_space, 'algo': tuning_algo, 'max_evals': max_trials, 'trials': hp_trials}
    if 'early_stop_fn' in tuning_params:
        try:
            train_module_name, early_stop_fn_name = tuning_params['early_stop_fn'].rsplit('.', 1)
        except ValueError:
            early_stop_fn_name = tuning_params['early_stop_fn']
            train_module_name = _USER_DEFINED_TRAIN_STEP_MODULE
        early_stop_fn = getattr(importlib.import_module(train_module_name), early_stop_fn_name)
        fmin_kwargs['early_stop_fn'] = early_stop_fn
    best_hp_params = fmin(**fmin_kwargs)
    best_hp_params = space_eval(search_space, best_hp_params)
    best_hp_estimator_loss = hp_trials.best_trial['result']['loss']
    if len(estimator_hardcoded_params) > 1:
        hardcoded_estimator_loss = objective(X_train, y_train, validation_df, estimator_hardcoded_params)
        if best_hp_estimator_loss < hardcoded_estimator_loss:
            best_hardcoded_params = {param_name: param_value for param_name, param_value in estimator_hardcoded_params.items() if param_name not in best_hp_params}
        else:
            best_hp_params = {}
            best_hardcoded_params = estimator_hardcoded_params
    else:
        best_hardcoded_params = {}
    return (best_hardcoded_params, best_hp_params)