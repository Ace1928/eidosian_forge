import collections
import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_state
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _MultiDeviceAddN(tensor_list, gradient_uid):
    """Adds tensors from potentially multiple devices."""
    tensors_on_device = collections.defaultdict(lambda: [])
    for tensor in tensor_list:
        tensors_on_device[tensor.device].append(tensor)
    summands = []

    def DeviceKey(dev):
        return '' if dev is None else dev
    for dev in sorted(tensors_on_device, key=DeviceKey):
        tensors = tensors_on_device[dev]
        with ops._colocate_with_for_gradient(tensors[0].op, gradient_uid, ignore_existing=True):
            summands.append(math_ops.add_n(tensors))
    return math_ops.add_n(summands)