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
def _create_op_internal(self, op_type, inputs, dtypes=None, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
    optimized_reduction_ops = {'Shape', 'Size', 'Rank', 'TensorListElementShape', 'TensorListLength'}
    if op_type in optimized_reduction_ops and (not util.output_all_intermediates()) and all((input.graph is self._forward_graph for input in inputs)) and all((_get_accumulator(input) is None for input in inputs)) and (not util_v1.GraphOrParentsInXlaContext(self._forward_graph)) and (not util.graph_wrapped_for_higher_order_tape_gradients(self._forward_graph)):
        return self._move_op_to_forward_graph(op_type, inputs, dtypes=dtypes, input_types=input_types, name=name, attrs=attrs, op_def=op_def, compute_device=compute_device)
    return super(_WhileBodyGradFuncGraph, self)._create_op_internal(op_type, inputs, dtypes=dtypes, input_types=input_types, name=name, attrs=attrs, op_def=op_def, compute_device=compute_device)