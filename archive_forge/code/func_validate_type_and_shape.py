import builtins
import datetime as dt
import importlib.util
import json
import string
import warnings
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
@classmethod
def validate_type_and_shape(cls, spec: str, value: Union[DataType, List[DataType], None], value_type: DataType, shape: Optional[Tuple[int, ...]]):
    """
        Validate that the value has the expected type and shape.
        """

    def _is_1d_array(value):
        return isinstance(value, (list, np.ndarray)) and np.array(value).ndim == 1
    if shape is None:
        return cls.enforce_param_datatype(f'{spec} with shape None', value, value_type)
    elif shape == (-1,):
        if not _is_1d_array(value):
            raise MlflowException.invalid_parameter_value(f'Value must be a 1D array with shape (-1,) for param {spec}, received {type(value).__name__} with ndim {np.array(value).ndim}')
        return [cls.enforce_param_datatype(f'{spec} internal values', v, value_type) for v in value]
    else:
        raise MlflowException.invalid_parameter_value(f'Shape must be None for scalar value or (-1,) for 1D array value for ParamSpec {spec}), received {shape}')