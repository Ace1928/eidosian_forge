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
def _get_tensor_shape(data, variable_dimension: Optional[int]=0) -> tuple:
    """Infer the shape of the inputted data.

    This method creates the shape of the tensor to store in the TensorSpec. The variable dimension
    is assumed to be the first dimension by default. This assumption can be overridden by inputting
    a different variable dimension or `None` to represent that the input tensor does not contain a
    variable dimension.

    Args:
        data: Dataset to infer from.
        variable_dimension: An optional integer representing a variable dimension.

    Returns:
        tuple: Shape of the inputted data (including a variable dimension)
    """
    from scipy.sparse import csc_matrix, csr_matrix
    if not isinstance(data, (np.ndarray, csr_matrix, csc_matrix)):
        raise TypeError(f"Expected numpy.ndarray or csc/csr matrix, got '{type(data)}'.")
    variable_input_data_shape = data.shape
    if variable_dimension is not None:
        try:
            variable_input_data_shape = list(variable_input_data_shape)
            variable_input_data_shape[variable_dimension] = -1
        except IndexError:
            raise MlflowException(f'The specified variable_dimension {variable_dimension} is out of bounds with respect to the number of dimensions {data.ndim} in the input dataset')
    return tuple(variable_input_data_shape)