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
def _convert_data_to_mlflow_dataset(data, targets=None, predictions=None):
    """Convert input data to mlflow dataset."""
    if 'pyspark' in sys.modules:
        from pyspark.sql import DataFrame as SparkDataFrame
    if isinstance(data, list):
        if not isinstance(data[0], (list, np.ndarray)):
            data = [[elm] for elm in data]
        return mlflow.data.from_numpy(np.array(data), targets=np.array(targets) if targets else None)
    elif isinstance(data, np.ndarray):
        return mlflow.data.from_numpy(data, targets=targets)
    elif isinstance(data, pd.DataFrame):
        return mlflow.data.from_pandas(df=data, targets=targets, predictions=predictions)
    elif 'pyspark' in sys.modules and isinstance(data, SparkDataFrame):
        return mlflow.data.from_spark(df=data, targets=targets)
    else:
        _logger.info(f'Cannot convert input data to `evaluate()` to an mlflow dataset, input must be a list, a numpy array, a panda Dataframe or a spark Dataframe, but received {type(data)}.')
        return data