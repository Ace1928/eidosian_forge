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
@RegisterPFor('Conv2DBackpropFilter')
def _convert_conv2d_backprop_filter(pfor_input):
    pfor_input.stack_inputs(stack_indices=[2])
    inputs, inputs_stacked, _ = pfor_input.input(0)
    filter_sizes = pfor_input.unstacked_input(1)
    grads = pfor_input.stacked_input(2)
    strides = pfor_input.get_attr('strides')
    padding = pfor_input.get_attr('padding')
    use_cudnn_on_gpu = pfor_input.get_attr('use_cudnn_on_gpu')
    data_format = pfor_input.get_attr('data_format')
    dilations = pfor_input.get_attr('dilations')
    if inputs_stacked:
        logging.warning('Conv2DBackpropFilter uses a while_loop. Fix that!')

        def while_body(i, ta):
            inp_i = inputs[i, ...]
            grad_i = grads[i, ...]
            output = nn_ops.conv2d_backprop_filter(inp_i, filter_sizes, grad_i, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format, dilations=dilations)
            return (i + 1, ta.write(i, output))
        n = array_ops.reshape(pfor_input.pfor.loop_len_vector, [])
        _, ta = while_loop.while_loop(lambda i, ta: i < n, while_body, (0, tensor_array_ops.TensorArray(inputs.dtype, n)))
        output = ta.stack()
        return wrap(output, True)
    else:
        grads, _, _ = _channel_flatten_input(grads, data_format)
        n = pfor_input.pfor.loop_len_vector
        old_filter_sizes = filter_sizes
        filter_sizes *= array_ops.concat([[1, 1, 1], n], axis=0)
        output = nn_ops.conv2d_backprop_filter(inputs, filter_sizes, grads, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format, dilations=dilations)
        new_filter_shape = array_ops.concat([old_filter_sizes[:3], n, [-1]], axis=0)
        output = array_ops.reshape(output, new_filter_shape)
        output = array_ops.transpose(output, [3, 0, 1, 2, 4])
        return wrap(output, True)