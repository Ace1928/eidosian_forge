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
def _concat_same_type(cls, to_concat: Sequence[Union['ArrowTensorArray', 'ArrowVariableShapedTensorArray']]) -> Union['ArrowTensorArray', 'ArrowVariableShapedTensorArray']:
    """
        Concatenate multiple tensor arrays.

        If one or more of the tensor arrays in to_concat are variable-shaped and/or any
        of the tensor arrays have a different shape than the others, a variable-shaped
        tensor array will be returned.
        """
    to_concat_types = [arr.type for arr in to_concat]
    if ArrowTensorType._need_variable_shaped_tensor_array(to_concat_types):
        return ArrowVariableShapedTensorArray.from_numpy([e for a in to_concat for e in a])
    else:
        storage = pa.concat_arrays([c.storage for c in to_concat])
        return ArrowTensorArray.from_storage(to_concat[0].type, storage)