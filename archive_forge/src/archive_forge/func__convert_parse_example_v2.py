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
@RegisterPFor('ParseExampleV2')
def _convert_parse_example_v2(pfor_input):
    serialized = pfor_input.stacked_input(0)
    sparse_keys = pfor_input.unstacked_input(2)
    dense_keys = pfor_input.unstacked_input(3)
    ragged_keys = pfor_input.unstacked_input(4)
    dense_defaults = [pfor_input.unstacked_input(i) for i in range(5, pfor_input.num_inputs)]
    num_sparse = pfor_input.get_attr('num_sparse')
    sparse_types = pfor_input.get_attr('sparse_types')
    ragged_value_types = pfor_input.get_attr('ragged_value_types')
    ragged_split_types = pfor_input.get_attr('ragged_split_types')
    dense_shapes = pfor_input.get_attr('dense_shapes')
    if serialized.shape.ndims not in (None, 1):
        raise ValueError(f'ParseExampleV2 can only be converted if `serialized` is scalar. Received shape: {serialized.shape}.')
    output = gen_parsing_ops.parse_example_v2(serialized=serialized, names=[], sparse_keys=sparse_keys, dense_keys=dense_keys, ragged_keys=ragged_keys, dense_defaults=dense_defaults, num_sparse=num_sparse, sparse_types=sparse_types, ragged_value_types=ragged_value_types, ragged_split_types=ragged_split_types, dense_shapes=dense_shapes)
    return [wrap(t, True, True) for t in nest.flatten(output)]