import json
import keyword
import logging
import math
import operator
import os
import pathlib
import signal
import struct
import sys
import urllib
import urllib.parse
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from decimal import Decimal
from types import FunctionType
from typing import Any, Dict, Optional
import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities import RunTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.validation import (
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _get_fully_qualified_class_name, insecure_hash
from mlflow.utils.annotations import developer_stable, experimental
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.string_utils import generate_feature_name_if_not_string
def _normalize_evaluators_and_evaluator_config_args(evaluators, evaluator_config):
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry

    def check_nesting_config_dict(_evaluator_name_list, _evaluator_name_to_conf_map):
        return isinstance(_evaluator_name_to_conf_map, dict) and all((k in _evaluator_name_list and isinstance(v, dict) for k, v in _evaluator_name_to_conf_map.items()))
    if evaluators is None:
        evaluator_name_list = list(_model_evaluation_registry._registry.keys())
        if len(evaluator_name_list) > 1:
            _logger.warning(f'Multiple registered evaluators are found {evaluator_name_list} and they will all be used in evaluation if they support the specified model type. If you want to evaluate with one evaluator, specify the `evaluator` argument and optionally specify the `evaluator_config` argument.')
        if evaluator_config is not None:
            conf_dict_value_error = MlflowException(message="If `evaluators` argument is None, all available evaluators will be used. If only the default evaluator is available, the `evaluator_config` argument is interpreted as the config dictionary for the default evaluator. Otherwise, the `evaluator_config` argument must be a dictionary mapping each evaluator's name to its own evaluator config dictionary.", error_code=INVALID_PARAMETER_VALUE)
            if evaluator_name_list == ['default']:
                if not isinstance(evaluator_config, dict):
                    raise conf_dict_value_error
                elif 'default' not in evaluator_config:
                    evaluator_name_to_conf_map = {'default': evaluator_config}
                else:
                    evaluator_name_to_conf_map = evaluator_config
            else:
                if not check_nesting_config_dict(evaluator_name_list, evaluator_config):
                    raise conf_dict_value_error
                evaluator_name_to_conf_map = evaluator_config
        else:
            evaluator_name_to_conf_map = {}
    elif isinstance(evaluators, str):
        if not (evaluator_config is None or isinstance(evaluator_config, dict)):
            raise MlflowException(message='If `evaluators` argument is the name of an evaluator, evaluator_config must be None or a dict containing config items for the evaluator.', error_code=INVALID_PARAMETER_VALUE)
        evaluator_name_list = [evaluators]
        evaluator_name_to_conf_map = {evaluators: evaluator_config}
    elif isinstance(evaluators, list):
        if evaluator_config is not None:
            if not check_nesting_config_dict(evaluators, evaluator_config):
                raise MlflowException(message='If `evaluators` argument is an evaluator name list, evaluator_config must be a dict contains mapping from evaluator name to individual evaluator config dict.', error_code=INVALID_PARAMETER_VALUE)
        evaluator_name_list = list(OrderedDict.fromkeys(evaluators))
        evaluator_name_to_conf_map = evaluator_config or {}
    else:
        raise MlflowException(message='`evaluators` argument must be None, an evaluator name string, or a list of evaluator names.', error_code=INVALID_PARAMETER_VALUE)
    return (evaluator_name_list, evaluator_name_to_conf_map)