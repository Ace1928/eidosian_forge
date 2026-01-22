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
def _chunk_tensor_arrays(cls, arrs: Sequence[Union['ArrowTensorArray', 'ArrowVariableShapedTensorArray']]) -> pa.ChunkedArray:
    """
        Create a ChunkedArray from multiple tensor arrays.
        """
    arrs_types = [arr.type for arr in arrs]
    if ArrowTensorType._need_variable_shaped_tensor_array(arrs_types):
        new_arrs = []
        for a in arrs:
            if isinstance(a.type, ArrowTensorType):
                a = a.to_variable_shaped_tensor_array()
            assert isinstance(a.type, ArrowVariableShapedTensorType)
            new_arrs.append(a)
        arrs = new_arrs
    return pa.chunked_array(arrs)