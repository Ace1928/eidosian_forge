import abc
import functools
from typing import Any, List, Optional, Sequence, Type
import warnings
import numpy as np
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal
from tensorflow.python.types import trace
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def __make_cmp_key(self, value):
    """Converts `value` to a hashable key."""
    if isinstance(value, (int, float, bool, np.generic, dtypes.DType, TypeSpec, tensor_shape.TensorShape)):
        return value
    if isinstance(value, compat.bytes_or_text_types):
        return value
    if value is None:
        return value
    if isinstance(value, dict):
        return tuple([tuple([self.__make_cmp_key(key), self.__make_cmp_key(value[key])]) for key in sorted(value.keys())])
    if isinstance(value, tuple):
        return tuple([self.__make_cmp_key(v) for v in value])
    if isinstance(value, list):
        return (list, tuple([self.__make_cmp_key(v) for v in value]))
    if isinstance(value, np.ndarray):
        return (np.ndarray, value.shape, TypeSpec.__nested_list_to_tuple(value.tolist()))
    raise ValueError(f'Cannot generate a hashable key for {self} because the _serialize() method returned an unsupproted value of type {type(value)}')