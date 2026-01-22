import collections
import enum
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import auto_control_deps_utils as utils
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import registry
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
def _get_resource_inputs(op):
    """Returns an iterable of resources touched by this `op`."""
    reads, writes = utils.get_read_write_resource_inputs(op)
    saturated = False
    while not saturated:
        saturated = True
        for key in _acd_resource_resolvers_registry.list():
            updated = _acd_resource_resolvers_registry.lookup(key)(op, reads, writes)
            if updated:
                reads = reads.difference(writes)
            saturated = saturated and (not updated)
    for t in reads:
        yield (t, ResourceType.READ_ONLY)
    for t in writes:
        yield (t, ResourceType.READ_WRITE)