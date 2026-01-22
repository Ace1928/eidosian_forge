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
def _is_contiguous_view(curr: np.ndarray, prev: Optional[np.ndarray]) -> bool:
    """Check if the provided tensor element is contiguous with the previous tensor
    element.

    Args:
        curr: The tensor element whose contiguity that we wish to check.
        prev: The previous tensor element in the tensor array.

    Returns:
        Whether the provided tensor element is contiguous with the previous tensor
        element.
    """
    if curr.base is None or not curr.data.c_contiguous or (prev is not None and curr.base is not prev.base):
        return False
    elif prev is not None and _get_buffer_address(curr) - _get_buffer_address(prev) != prev.base.dtype.itemsize * prev.size:
        return False
    else:
        return True