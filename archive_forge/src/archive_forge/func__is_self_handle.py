import collections
import contextlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _is_self_handle(self, x):
    """Check if the tensor `x` is the same Mutex as `self._handle`."""
    if isinstance(x, ops.EagerTensor):
        return x is self._handle
    return x.op.type == 'MutexV2' and x.op.get_attr('shared_name') and (x.op.get_attr('shared_name') == self._handle.op.get_attr('shared_name')) and (x.op.device == self._handle.op.device or _get_colocation(x.op) == _get_colocation(self._handle.op))