import collections
from functools import partial
import string
import sys
import traceback
import numpy as np
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
@RegisterPFor('FusedBatchNormGradV3')
def _convert_fused_batch_norm_grad(pfor_input):
    pfor_input.stack_inputs()
    data_format = pfor_input.get_attr('data_format')
    y_backprop = pfor_input.stacked_input(0)
    y_backprop, _, _ = _channel_flatten_input(y_backprop, data_format)
    x = pfor_input.stacked_input(1)
    x, x_reverse_order, x_reverse_shape = _channel_flatten_input(x, data_format)
    inputs = [y_backprop, x] + [array_ops.reshape(pfor_input.stacked_input(i), [-1]) for i in range(2, pfor_input.num_inputs)]
    outputs = _create_op(pfor_input.op_type, inputs, [x.dtype for x in pfor_input.outputs], attrs=pfor_input.op.node_def.attr).outputs
    x_backprop = outputs[0]
    x_backprop = array_ops.reshape(x_backprop, x_reverse_shape)
    x_backprop = array_ops.transpose(x_backprop, x_reverse_order)
    n = pfor_input.pfor.loop_len_vector
    outputs = [_unflatten_first_dim(x, n) for x in outputs[1:]]
    outputs = [x_backprop] + outputs
    return [wrap(output, True) for output in outputs]