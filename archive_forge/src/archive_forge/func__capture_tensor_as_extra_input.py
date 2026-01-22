import collections
import hashlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
def _capture_tensor_as_extra_input(self, tensor, name=None):
    self.extra_inputs.append(tensor)
    with ops.control_dependencies(None):
        ph = array_ops.placeholder(tensor.dtype, shape=tensor.get_shape(), name=name)
    if isinstance(tensor, ops.EagerTensor):
        handle_data = tensor._handle_data
        if handle_data:
            handle_data = handle_data.SerializeToString()
    else:
        with tensor.graph._c_graph.get() as c_graph:
            handle_data = c_api.GetHandleShapeAndType(c_graph, tensor._as_tf_output())
    if handle_data:
        with ph.graph._c_graph.get() as c_graph:
            c_api.SetHandleShapeAndType(c_graph, ph._as_tf_output(), compat.as_bytes(handle_data))
    self.inputs.append(ph)
    self._captured[tensor.ref()] = ph
    self.extra_args.append(ph)
    if _is_guaranteed_const(tensor):
        with ops.control_dependencies(None):
            return array_ops.guarantee_const(ph)
    else:
        return ph