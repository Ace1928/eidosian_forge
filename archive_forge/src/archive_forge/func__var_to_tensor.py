import copy
import math
from typing import Sequence
import weakref
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _var_to_tensor(var, dtype=None, name=None, as_ref=False):
    """Converts a `ShardedVariable` to a `Tensor`."""
    del name
    if dtype is not None and (not dtype.is_compatible_with(var.dtype)):
        raise ValueError('Incompatible type conversion requested to type {!r} for variable of type {!r}'.format(dtype.name, var.dtype.name))
    if as_ref:
        raise NotImplementedError("ShardedVariable doesn't support being used as a reference.")
    if 'embedding_lookup' in ops.get_name_scope():
        raise TypeError('Converting ShardedVariable to tensor in embedding lookup ops is disallowed.')
    return array_ops.concat(var.variables, axis=0)