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
def _validate_is_dict(d):
    if not isinstance(d, dict):
        raise MlflowException(f'Expected each item in example to be dict, got {type(d).__name__}', INVALID_PARAMETER_VALUE)