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
def _validate_input_dictionary_contains_only_strings_and_lists_of_strings(data) -> None:
    invalid_keys = [key for key in data.keys() if not isinstance(key, (str, int)) or isinstance(key, bool)]
    if invalid_keys:
        raise MlflowException(f'The dictionary keys are not all strings or indexes. Invalid keys: {invalid_keys}')
    if any((isinstance(value, np.ndarray) for value in data.values())) and (not all((isinstance(value, np.ndarray) for value in data.values()))):
        raise MlflowException('The dictionary values are not all numpy.ndarray.')
    invalid_values = [key for key, value in data.items() if isinstance(value, list) and (not all((isinstance(item, (str, bytes)) for item in value))) or not isinstance(value, (np.ndarray, list, str, bytes))]
    if invalid_values:
        raise MlflowException.invalid_parameter_value(f'Invalid values in dictionary. If passing a dictionary containing strings, all values must be either strings or lists of strings. If passing a dictionary containing numeric values, the data must be enclosed in a numpy.ndarray. The following keys in the input dictionary are invalid: {invalid_values}')