import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import DataType
from mlflow.types.schema import (
def _infer_colspec_type(data: Any) -> Union[DataType, Array, Object]:
    """
    Infer an MLflow Colspec type from the dataset.

    Args:
        data: data to infer from.

    Returns:
        Object
    """
    dtype = _infer_datatype(data)
    if dtype is None:
        raise MlflowException('A column of nested array type must include at least one non-empty array.')
    return dtype