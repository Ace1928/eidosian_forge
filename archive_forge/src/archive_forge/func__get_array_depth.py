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
def _get_array_depth(l: Any) -> int:
    if isinstance(l, np.ndarray):
        return l.ndim
    if isinstance(l, list):
        return max((_get_array_depth(item) for item in l)) + 1 if l else 1
    return 0