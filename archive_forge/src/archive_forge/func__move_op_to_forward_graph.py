import collections
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util_v1
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
def _move_op_to_forward_graph(self, op_type, inputs, dtypes=None, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
    if not hasattr(self._forward_graph, '_optimized_reduction_ops_cache'):
        self._forward_graph._optimized_reduction_ops_cache = {}
    cache_key = self._get_optimized_reduction_ops_cache_key(op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)
    cached_op = self._forward_graph._optimized_reduction_ops_cache.get(cache_key)
    if cached_op is not None:
        return cached_op
    with self._forward_graph.as_default():
        name = ops.name_from_scope_name(name)
        result = self._forward_graph._create_op_internal(op_type, inputs, dtypes=dtypes, input_types=input_types, name=name, attrs=attrs, op_def=op_def, compute_device=compute_device)
        self._forward_graph._optimized_reduction_ops_cache[cache_key] = result
        return result