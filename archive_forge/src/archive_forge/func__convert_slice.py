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
@RegisterPFor('Slice')
def _convert_slice(pfor_input):
    t = pfor_input.stacked_input(0)
    begin, begin_stacked, _ = pfor_input.input(1)
    size = pfor_input.unstacked_input(2)
    if not begin_stacked:
        begin = array_ops.concat([[0], begin], axis=0)
        size = array_ops.concat([[-1], size], axis=0)
        return wrap(array_ops.slice(t, begin, size), True)
    else:
        t_shape = array_ops.shape(t)
        size = math_ops.cast(size, t_shape.dtype)
        begin = math_ops.cast(begin, t_shape.dtype)
        n = math_ops.cast(pfor_input.pfor.loop_len_vector, t_shape.dtype)
        original_unstacked_shape = _stack(t_shape[1:], n).t
        broadcast_size = _stack(size, n).t
        result_shape = array_ops.where(math_ops.less(broadcast_size, 0), original_unstacked_shape - begin + broadcast_size + 1, broadcast_size)
        result_shape = math_ops.cast(math_ops.reduce_max(result_shape, axis=0), dtypes.int64)
        cumsize = math_ops.cumprod(result_shape, exclusive=True, reverse=True)
        result_num_elements = math_ops.reduce_prod(result_shape)
        result_base_coordinates = math_ops.range(result_num_elements, dtype=dtypes.int64)[:, None] // cumsize[None, :] % result_shape[None, :]
        result_coordinates = begin[:, None, :] + math_ops.cast(result_base_coordinates, begin.dtype)[None, :, :]
        result_flat = array_ops.gather_nd(params=t, indices=result_coordinates, batch_dims=1)
        result_stacked_shape = array_ops.concat([math_ops.cast(pfor_input.pfor.loop_len_vector, result_shape.dtype), result_shape], axis=0)
        return wrap(array_ops.reshape(result_flat, result_stacked_shape), True)