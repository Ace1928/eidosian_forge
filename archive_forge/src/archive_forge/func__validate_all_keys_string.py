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
def _validate_all_keys_string(d):
    keys = list(d.keys())
    if not _is_all_string(keys):
        raise MlflowException(f'Expected example to be dict with string keys, got {keys}', INVALID_PARAMETER_VALUE)