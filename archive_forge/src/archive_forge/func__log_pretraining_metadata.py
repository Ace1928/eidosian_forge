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
def _log_pretraining_metadata(estimator, params, input_df):
    if params and isinstance(params, dict):
        estimator = estimator.copy(params)
    autologging_metadata = _gen_estimator_metadata(estimator)
    artifact_dict = {}
    param_map = _get_instance_param_map(estimator, autologging_metadata.uid_to_indexed_name_map)
    if _should_log_hierarchy(estimator):
        artifact_dict['hierarchy'] = autologging_metadata.hierarchy
    for param_search_estimator in autologging_metadata.param_search_estimators:
        param_search_estimator_name = f'{autologging_metadata.uid_to_indexed_name_map[param_search_estimator.uid]}'
        artifact_dict[param_search_estimator_name] = {}
        artifact_dict[param_search_estimator_name]['tuning_parameter_map_list'] = _get_tuning_param_maps(param_search_estimator, autologging_metadata.uid_to_indexed_name_map)
        artifact_dict[param_search_estimator_name]['tuned_estimator_parameter_map'] = _get_instance_param_map_recursively(param_search_estimator.getEstimator(), 1, autologging_metadata.uid_to_indexed_name_map)
    if artifact_dict:
        mlflow.log_dict(artifact_dict, artifact_file='estimator_info.json')
    _log_estimator_params(param_map)
    mlflow.set_tags(_get_estimator_info_tags(estimator))
    if log_datasets:
        try:
            context_tags = context_registry.resolve_tags()
            code_source = CodeDatasetSource(context_tags)
            dataset = SparkDataset(df=input_df, source=code_source)
            mlflow.log_input(dataset, 'train')
        except Exception as e:
            _logger.warning('Failed to log training dataset information to MLflow Tracking. Reason: %s', e)