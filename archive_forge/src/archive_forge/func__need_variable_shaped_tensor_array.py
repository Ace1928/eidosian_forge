import itertools
import json
import sys
from typing import Iterable, Optional, Tuple, List, Sequence, Union
from pkg_resources._vendor.packaging.version import parse as parse_version
import numpy as np
import pyarrow as pa
from ray.air.util.tensor_extensions.utils import (
from ray._private.utils import _get_pyarrow_version
from ray.util.annotations import PublicAPI
@classmethod
def _need_variable_shaped_tensor_array(cls, array_types: Sequence[Union['ArrowTensorType', 'ArrowVariableShapedTensorType']]) -> bool:
    """
        Whether the provided list of tensor types needs a variable-shaped
        representation (i.e. `ArrowVariableShapedTensorType`) when concatenating
        or chunking. If one or more of the tensor types in `array_types` are
        variable-shaped and/or any of the tensor arrays have a different shape
        than the others, a variable-shaped tensor array representation will be
        required and this method will return True.

        Args:
            array_types: List of tensor types to check if a variable-shaped
            representation is required for concatenation

        Returns:
            True if concatenating arrays with types `array_types` requires
            a variable-shaped representation
        """
    shape = None
    for arr_type in array_types:
        if isinstance(arr_type, ArrowVariableShapedTensorType):
            return True
        if not isinstance(arr_type, ArrowTensorType):
            raise ValueError(f'All provided array types must be an instance of either ArrowTensorType or ArrowVariableShapedTensorType, but got {arr_type}')
        if shape is not None and arr_type.shape != shape:
            return True
        shape = arr_type.shape
    return False