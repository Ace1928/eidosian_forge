import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _iter_slices(full_shape, num_slices, slice_dim):
    """Slices a given a shape along the specified dimension."""
    num_slices_with_excess = full_shape[slice_dim] % num_slices
    offset = [0] * len(full_shape)
    min_slice_len = full_shape[slice_dim] // num_slices
    for i in range(num_slices):
        shape = full_shape[:]
        shape[slice_dim] = min_slice_len + bool(i < num_slices_with_excess)
        yield (offset[:], shape)
        offset[slice_dim] += shape[slice_dim]