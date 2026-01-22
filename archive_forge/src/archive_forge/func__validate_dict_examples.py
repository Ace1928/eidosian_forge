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
def _validate_dict_examples(examples, num_items=None):
    examples_iter = iter(examples)
    first_example = next(examples_iter)
    _validate_is_dict(first_example)
    _validate_has_items(first_example)
    if num_items is not None:
        _validate_num_items(first_example, num_items)
    _validate_all_keys_string(first_example)
    _validate_all_values_string(first_example)
    first_keys = first_example.keys()
    for example in examples_iter:
        _validate_is_dict(example)
        _validate_has_items(example)
        if num_items is not None:
            _validate_num_items(example, num_items)
        _validate_all_keys_string(example)
        _validate_all_values_string(example)
        _validate_keys_match(example, first_keys)