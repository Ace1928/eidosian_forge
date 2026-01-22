import functools
import numbers
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.platform import device_context
from tensorflow.python.platform import build_info
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@ops.RegisterStatistics('Conv2D', 'flops')
def _calc_conv_flops(graph, node):
    """Calculates the compute resources needed for Conv2D."""
    input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    input_shape.assert_is_fully_defined()
    filter_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[1])
    filter_shape.assert_is_fully_defined()
    output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    output_shape.assert_is_fully_defined()
    filter_height = int(filter_shape[0])
    filter_width = int(filter_shape[1])
    filter_in_depth = int(filter_shape[2])
    output_count = np.prod(output_shape.as_list(), dtype=np.int64)
    return ops.OpStats('flops', output_count * filter_in_depth * filter_height * filter_width * 2)