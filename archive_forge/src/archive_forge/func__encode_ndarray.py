from __future__ import annotations
import logging # isort:skip
import base64
import datetime as dt
import sys
from array import array as TypedArray
from math import isinf, isnan
from types import SimpleNamespace
from typing import (
import numpy as np
from ..util.dataclasses import (
from ..util.dependencies import uses_pandas
from ..util.serialization import (
from ..util.warnings import BokehUserWarning, warn
from .types import ID
def _encode_ndarray(self, obj: npt.NDArray[Any]) -> NDArrayRep:
    array = transform_array(obj)
    data: ArrayRepLike | BytesRep
    dtype: NDDataType
    if array_encoding_disabled(array):
        data = self._encode_list(array.flatten().tolist())
        dtype = 'object'
    else:
        data = self._encode_bytes(array.data)
        dtype = cast(NDDataType, array.dtype.name)
    return NDArrayRep(type='ndarray', array=data, shape=list(array.shape), dtype=dtype, order=sys.byteorder)