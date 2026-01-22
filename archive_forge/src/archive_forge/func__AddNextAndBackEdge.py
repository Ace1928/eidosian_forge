import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _AddNextAndBackEdge(m, v, enforce_shape_invariant=True):
    """Add NextIteration and back edge from v to m."""
    if isinstance(m, tensor_lib.Tensor):
        v = ops.convert_to_tensor(v)
        v = _NextIteration(v)
        if enforce_shape_invariant:
            _EnforceShapeInvariant(m, v)
        m.op._update_input(1, v)
    elif isinstance(m, composite_tensor.CompositeTensor):

        def update_component(m_component, v_component):
            m_component.op._update_input(1, v_component)
        if isinstance(m, indexed_slices.IndexedSlices):
            v = math_ops._as_indexed_slices(v, optimize=False)
        v = _NextIteration(v)
        return nest.map_structure(update_component, m, v, expand_composites=True)
    else:
        raise TypeError(f"'m' must be a Tensor or CompositeTensor. Received: {type(m)}.")
    return v