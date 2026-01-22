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
def _read_log_model_allowlist():
    """
    Reads the module allowlist and returns it as a set.
    """
    from mlflow.utils._spark_utils import _get_active_spark_session
    if sys.version_info.major > 2 and sys.version_info.minor > 8:
        from importlib.resources import as_file, files
        with as_file(files(__name__).joinpath('log_model_allowlist.txt')) as file:
            builtin_allowlist_file = file.as_posix()
    else:
        from importlib.resources import path
        with path(__name__, 'log_model_allowlist.txt') as file:
            builtin_allowlist_file = file.as_posix()
    spark_session = _get_active_spark_session()
    if not spark_session:
        _logger.info('No SparkSession detected. Autologging will log pyspark.ml models contained in the default allowlist. To specify a custom allowlist, initialize a SparkSession prior to calling mlflow.pyspark.ml.autolog() and specify the path to your allowlist file via the spark.mlflow.pysparkml.autolog.logModelAllowlistFile conf.')
        return _read_log_model_allowlist_from_file(builtin_allowlist_file)
    allowlist_file = spark_session.sparkContext._conf.get('spark.mlflow.pysparkml.autolog.logModelAllowlistFile', None)
    if allowlist_file:
        try:
            return _read_log_model_allowlist_from_file(allowlist_file)
        except Exception:
            _logger.exception('Reading from custom log_models allowlist file %s failed, fallback to built-in allowlist file.', allowlist_file)
            return _read_log_model_allowlist_from_file(builtin_allowlist_file)
    else:
        return _read_log_model_allowlist_from_file(builtin_allowlist_file)