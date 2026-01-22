import json
import logging
import os
import sys
import traceback
import weakref
from collections import OrderedDict, defaultdict, namedtuple
from itertools import zip_longest
from urllib.parse import urlparse
import numpy as np
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.spark_dataset import SparkDataset
from mlflow.entities import Metric, Param
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import (
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.os import is_windows
from mlflow.utils.rest_utils import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
class _AutologgingMetricsManager:
    """
    This class is designed for holding information which is used by autologging metrics
    It will hold information of:
    (1) a map of "prediction result object id" to a tuple of dataset name(the dataset is
       the one which generate the prediction result) and run_id.
       Note: We need this map instead of setting the run_id into the "prediction result object"
       because the object maybe a numpy array which does not support additional attribute
       assignment.
    (2) _log_post_training_metrics_enabled flag, in the following method scope:
       `Estimator.fit`, `Model.transform`, `Evaluator.evaluate`,
       in order to avoid nested/duplicated autologging metric, when run into these scopes,
       we need temporarily disable the metric autologging.
    (3) _eval_dataset_info_map, it is a double level map:
       `_eval_dataset_info_map[run_id][eval_dataset_var_name]` will get a list, each
       element in the list is an id of "eval_dataset" instance.
       This data structure is used for:
        * generating unique dataset name key when autologging metric. For each eval dataset object,
          if they have the same eval_dataset_var_name, but object ids are different,
          then they will be assigned different name (via appending index to the
          eval_dataset_var_name) when autologging.
    (4) _evaluator_call_info, it is a double level map:
       `_metric_api_call_info[run_id][metric_name]` wil get a list of tuples, each tuple is:
         (logged_metric_key, evaluator_information)
        Evaluator information includes evaluator class name and params, these information
        will also be logged into "metric_info.json" artifacts.

    Note: this class is not thread-safe.
    Design rule for this class:
     Because this class instance is a global instance, in order to prevent memory leak, it should
     only holds IDs and other small objects references. This class internal data structure should
     avoid reference to user dataset variables or model variables.
    """

    def __init__(self):
        self._pred_result_id_to_dataset_name_and_run_id = {}
        self._eval_dataset_info_map = defaultdict(lambda: defaultdict(list))
        self._evaluator_call_info = defaultdict(lambda: defaultdict(list))
        self._log_post_training_metrics_enabled = True
        self._metric_info_artifact_need_update = defaultdict(lambda: False)

    def should_log_post_training_metrics(self):
        """
        Check whether we should run patching code for autologging post training metrics.
        This checking should surround the whole patched code due to the safe guard checking,
        See following note.

        Note: It includes checking `_SparkTrainingSession.is_active()`, This is a safe guarding
        for meta-estimator (e.g. CrossValidator/TrainValidationSplit) case:
          running CrossValidator.fit, the nested `estimator.fit` will be called in parallel,
          but, the _autolog_training_status is a global status without thread-safe lock protecting.
          This safe guarding will prevent code run into this case.
        """
        return not _SparkTrainingSession.is_active() and self._log_post_training_metrics_enabled

    def disable_log_post_training_metrics(self):

        class LogPostTrainingMetricsDisabledScope:

            def __enter__(inner_self):
                inner_self.old_status = self._log_post_training_metrics_enabled
                self._log_post_training_metrics_enabled = False

            def __exit__(inner_self, exc_type, exc_val, exc_tb):
                self._log_post_training_metrics_enabled = inner_self.old_status
        return LogPostTrainingMetricsDisabledScope()

    @staticmethod
    def get_run_id_for_model(model):
        return getattr(model, '_mlflow_run_id', None)

    @staticmethod
    def is_metric_value_loggable(metric_value):
        """
        check whether the specified `metric_value` is a numeric value which can be logged
        as an MLflow metric.
        """
        return isinstance(metric_value, (int, float, np.number)) and (not isinstance(metric_value, bool))

    def register_model(self, model, run_id):
        """
        In `patched_fit`, we need register the model with the run_id used in `patched_fit`
        So that in following metric autologging, the metric will be logged into the registered
        run_id
        """
        model._mlflow_run_id = run_id

    @staticmethod
    def gen_name_with_index(name, index):
        assert index >= 0
        if index == 0:
            return name
        else:
            return f'{name}-{index + 1}'

    def register_prediction_input_dataset(self, model, eval_dataset):
        """
        Register prediction input dataset into eval_dataset_info_map, it will do:
         1. inspect eval dataset var name.
         2. check whether eval_dataset_info_map already registered this eval dataset.
            will check by object id.
         3. register eval dataset with id.
         4. return eval dataset name with index.

        Note: this method include inspecting argument variable name.
         So should be called directly from the "patched method", to ensure it capture
         correct argument variable name.
        """
        eval_dataset_name = _inspect_original_var_name(eval_dataset, fallback_name='unknown_dataset')
        eval_dataset_id = id(eval_dataset)
        run_id = self.get_run_id_for_model(model)
        registered_dataset_list = self._eval_dataset_info_map[run_id][eval_dataset_name]
        for i, id_i in enumerate(registered_dataset_list):
            if eval_dataset_id == id_i:
                index = i
                break
        else:
            index = len(registered_dataset_list)
        if index == len(registered_dataset_list):
            registered_dataset_list.append(eval_dataset_id)
        return self.gen_name_with_index(eval_dataset_name, index)

    def register_prediction_result(self, run_id, eval_dataset_name, predict_result):
        """
        Register the relationship
         id(prediction_result) --> (eval_dataset_name, run_id)
        into map `_pred_result_id_to_dataset_name_and_run_id`
        """
        value = (eval_dataset_name, run_id)
        prediction_result_id = id(predict_result)
        self._pred_result_id_to_dataset_name_and_run_id[prediction_result_id] = value

        def clean_id(id_):
            _AUTOLOGGING_METRICS_MANAGER._pred_result_id_to_dataset_name_and_run_id.pop(id_, None)
        weakref.finalize(predict_result, clean_id, prediction_result_id)

    def get_run_id_and_dataset_name_for_evaluator_call(self, pred_result_dataset):
        """
        Given a registered prediction result dataset object,
        return a tuple of (run_id, eval_dataset_name)
        """
        if id(pred_result_dataset) in self._pred_result_id_to_dataset_name_and_run_id:
            dataset_name, run_id = self._pred_result_id_to_dataset_name_and_run_id[id(pred_result_dataset)]
            return (run_id, dataset_name)
        else:
            return (None, None)

    def gen_evaluator_info(self, evaluator):
        """
        Generate evaluator information, include evaluator class name and params.
        """
        class_name = _get_fully_qualified_class_name(evaluator)
        param_map = _truncate_dict(_get_param_map(evaluator), MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)
        return {'evaluator_class': class_name, 'params': param_map}

    def register_evaluator_call(self, run_id, metric_name, dataset_name, evaluator_info):
        """
        Register the `Evaluator.evaluate` call, including register the evaluator information
        (See doc of `gen_evaluator_info` method) into the corresponding run_id and metric_name
        entry in the registry table.
        """
        evaluator_call_info_list = self._evaluator_call_info[run_id][metric_name]
        index = len(evaluator_call_info_list)
        metric_name_with_index = self.gen_name_with_index(metric_name, index)
        metric_key = f'{metric_name_with_index}_{dataset_name}'
        evaluator_call_info_list.append((metric_key, evaluator_info))
        self._metric_info_artifact_need_update[run_id] = True
        return metric_key

    def log_post_training_metric(self, run_id, key, value):
        """
        Log the metric into the specified mlflow run.
        and it will also update the metric_info artifact if needed.
        """
        client = MlflowClient()
        client.log_metric(run_id=run_id, key=key, value=value)
        if self._metric_info_artifact_need_update[run_id]:
            evaluator_call_list = []
            for v in self._evaluator_call_info[run_id].values():
                evaluator_call_list.extend(v)
            evaluator_call_list.sort(key=lambda x: x[0])
            dict_to_log = OrderedDict(evaluator_call_list)
            client.log_dict(run_id=run_id, dictionary=dict_to_log, artifact_file='metric_info.json')
            self._metric_info_artifact_need_update[run_id] = False