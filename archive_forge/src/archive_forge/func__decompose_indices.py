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
def _decompose_indices(self, indices):
    """Decompose a global 1D indices into a list of per-variable indices."""
    if indices.shape.rank != 1:
        raise ValueError(f'ShardedVariable: indices must be 1D Tensor for sparse operations. Received shape: {indices.shape}')
    base = self._shape[0] // len(self._variables)
    extra = self._shape[0] % len(self._variables)
    expect_first_dim = [base] * len(self._variables)
    for i in range(extra):
        expect_first_dim[i] = expect_first_dim[i] + 1
    actual_first_dim = [v.shape.as_list()[0] for v in self._variables]
    if expect_first_dim != actual_first_dim:
        raise NotImplementedError('scater_xxx ops are not supported in ShardedVariale that does not conform to "div" sharding')
    partition_assignments = math_ops.maximum(indices // (base + 1), (indices - extra) // base)
    local_indices = array_ops.where(partition_assignments < extra, indices % (base + 1), (indices - extra) % base)
    partition_assignments = math_ops.cast(partition_assignments, dtypes.int32)
    per_var_indices = data_flow_ops.dynamic_partition(local_indices, partition_assignments, len(self._variables))
    return (per_var_indices, partition_assignments)