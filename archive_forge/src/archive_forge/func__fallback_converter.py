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
def _fallback_converter(pfor_input, root_cause='', warn=False):
    msg = f'Using a while_loop for converting {pfor_input.op_type} cause {root_cause}'
    if warn:
        logging.warning(msg)
    else:
        logging.debug(msg)
    output_dtypes = [x.dtype for x in pfor_input.outputs]
    iter_vec = pfor_input.pfor.loop_len_vector
    iter_vec_value = tensor_util.constant_value(iter_vec)
    if iter_vec_value is not None:
        iters = iter_vec_value[0].item()
    else:
        iters = iter_vec[0]

    def while_body(i, *ta_list):
        """Body of while loop."""
        inputs = [x[i, ...] if stacked else x for x, stacked, _ in pfor_input.inputs]
        op_outputs = _create_op(pfor_input.op_type, inputs, output_dtypes, attrs=pfor_input.op.node_def.attr).outputs
        outputs = []
        for out, ta in zip(op_outputs, ta_list):
            assert isinstance(out, tensor_lib.Tensor)
            outputs.append(ta.write(i, out))
        return tuple([i + 1] + outputs)
    ta_list = while_loop.while_loop(lambda i, *ta: i < iters, while_body, [0] + [tensor_array_ops.TensorArray(dtype, iters) for dtype in output_dtypes])[1:]
    return tuple([wrap(ta.stack(), True) for ta in ta_list])