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
@RegisterPFor('Gather')
@RegisterPFor('GatherV2')
def _convert_gather(pfor_input):
    param, param_stacked, _ = pfor_input.input(0)
    indices, indices_stacked, _ = pfor_input.input(1)
    batch_dims = pfor_input.get_attr('batch_dims')
    op_type = pfor_input.op_type
    if op_type == 'Gather':
        validate_indices = pfor_input.get_attr('validate_indices')
        axis = 0
    else:
        validate_indices = None
        axis = math_ops.cast(pfor_input.unstacked_input(2), dtypes.int32)
        axis_value = tensor_util.constant_value(axis)
        if axis_value is not None:
            axis = axis_value
    if indices_stacked and (not param_stacked):
        if indices is pfor_input.pfor.all_indices and axis == 0:
            param_shape0 = tensor_shape.dimension_value(param.shape[0])
            indices_shape0 = tensor_shape.dimension_value(indices.shape[0])
            if param_shape0 is not None and indices_shape0 == param_shape0:
                return wrap(param, True)
        if batch_dims != 0:
            batch_dims_pos = batch_dims
            if batch_dims < 0:
                batch_dims_pos += array_ops.rank(indices)
            order = array_ops.concat([math_ops.range(1, batch_dims_pos + 1), [0], math_ops.range(batch_dims_pos + 1, array_ops.rank(indices))], axis=0)
            indices = array_ops.transpose(indices, order)
        output = array_ops.gather(param, indices, validate_indices=validate_indices, axis=axis, batch_dims=batch_dims)
        if axis != 0:
            axis = smart_cond.smart_cond(axis < 0, lambda: axis + array_ops.rank(param), lambda: ops.convert_to_tensor(axis))
            order = array_ops.concat([[axis], math_ops.range(axis), math_ops.range(axis + 1, array_ops.rank(output))], axis=0)
            output = smart_cond.smart_cond(math_ops.equal(axis, 0), lambda: output, lambda: array_ops.transpose(output, order))
        return wrap(output, True)
    if param_stacked:
        pfor_input.stack_inputs(stack_indices=[1])
        indices = pfor_input.stacked_input(1)
        if isinstance(axis, tensor_lib.Tensor):
            axis = array_ops.where(axis >= 0, axis + 1, axis)
        else:
            axis = axis + 1 if axis >= 0 else axis
        batch_dims = batch_dims + 1 if batch_dims >= 0 else batch_dims
        output = array_ops.gather(param, indices, axis=axis, batch_dims=batch_dims)
        return wrap(output, True)