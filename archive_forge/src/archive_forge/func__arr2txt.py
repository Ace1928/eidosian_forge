from http://www.nitrc.org/projects/gifti/
from __future__ import annotations
import base64
import sys
import warnings
from copy import copy
from typing import Type, cast
import numpy as np
from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
from .parse_gifti_fast import GiftiImageParser
def _arr2txt(arr, elem_fmt):
    arr = np.asarray(arr)
    assert arr.dtype.names is None
    if arr.ndim == 1:
        arr = arr[:, None]
    fmt = ' '.join([elem_fmt] * arr.shape[1])
    return '\n'.join((fmt % tuple(row) for row in arr))