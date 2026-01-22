import json
import logging
from functools import cached_property
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import (
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.schema import TensorDatasetSchema
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema
@staticmethod
def _get_schema_from_tf_dataset_dict_numpy_data(numpy_data: Dict[Any, Any]) -> Schema:
    if not all((isinstance(data_element, np.ndarray) for data_element in numpy_data.values())):
        raise MlflowException('Failed to infer schema for tf.data.Dataset. Schemas can only be inferred if the dataset consists of tensors. Ragged tensors, tensor arrays, and other types are not supported. Additionally, datasets with nested tensors are not supported.', INVALID_PARAMETER_VALUE)
    return _infer_schema(numpy_data)