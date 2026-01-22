import threading
import weakref
from tensorflow.python import _pywrap_parallel_device
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def _unpack_tensor(self, parallel_tensor):
    """Helper to unpack a single tensor."""
    if not isinstance(parallel_tensor, (tensor_lib.Tensor, composite_tensor.CompositeTensor, variables.Variable)):
        raise ValueError('Expected a tensor, got {}.'.format(parallel_tensor))
    with ops.device(self._name):
        return tpu_ops.tpu_replicated_output(parallel_tensor, num_replicas=len(self.components))