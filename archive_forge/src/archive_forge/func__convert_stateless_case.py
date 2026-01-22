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
@RegisterPFor('Case')
@RegisterPFor('StatelessCase')
def _convert_stateless_case(pfor_input):
    branch_idx, is_stacked, _ = pfor_input.input(0)
    branches = pfor_input.get_attr('branches')
    inputs = pfor_input.inputs[1:]
    if is_stacked:
        logging.info('Running stacked flow')
        switch_indices = data_flow_ops.dynamic_partition(pfor_input.pfor.all_indices, branch_idx, len(branches))
        if pfor_input.pfor.all_indices_partitioned:
            partitioned_indices = data_flow_ops.dynamic_partition(math_ops.range(pfor_input.pfor.loop_len_vector[0]), branch_idx, len(branches))
        else:
            partitioned_indices = switch_indices
        input_list = []
        for indices in partitioned_indices:
            input_list.append(_partition_inputs_for_indices(inputs, indices))
        outputs = []
        for b, indices, inputs in zip(branches, switch_indices, input_list):
            out = _outputs_for_branch(b.name, indices, pfor_input, inputs)
            outputs.extend(out)
        out = data_flow_ops.dynamic_stitch(partitioned_indices, outputs)
        return [wrap(out, True)]
    else:
        new_branches = []
        for b in branches:

            def new_function(func=b.name):
                return _outputs_for_branch(func, None, pfor_input, pfor_input.inputs[1:])
            new_branches.append(new_function)
        outputs = []
        outputs = control_flow_switch_case.switch_case(branch_idx, new_branches)
        return [wrap(t, True) for t in outputs]