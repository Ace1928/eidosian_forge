import abc
import itertools
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_model
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import save as save_model
from tensorflow.python.util import nest
def _test_matmul_base(self, use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg, with_batch):
    if not with_batch and len(shapes_info.shape) <= 2:
        return
    with self.session(graph=ops.Graph()) as sess:
        sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
        operator, mat = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
        x = self.make_x(operator, adjoint=adjoint, with_batch=with_batch)
        if adjoint_arg:
            op_matmul = operator.matmul(linalg.adjoint(x), adjoint=adjoint, adjoint_arg=adjoint_arg)
        else:
            op_matmul = operator.matmul(x, adjoint=adjoint)
        mat_matmul = math_ops.matmul(mat, x, adjoint_a=adjoint)
        if not use_placeholder:
            self.assertAllEqual(op_matmul.shape, mat_matmul.shape)
        if blockwise_arg and len(operator.operators) > 1:
            block_dimensions = operator._block_range_dimensions() if adjoint else operator._block_domain_dimensions()
            block_dimensions_fn = operator._block_range_dimension_tensors if adjoint else operator._block_domain_dimension_tensors
            split_x = linear_operator_util.split_arg_into_blocks(block_dimensions, block_dimensions_fn, x, axis=-2)
            if adjoint_arg:
                split_x = [linalg.adjoint(y) for y in split_x]
            split_matmul = operator.matmul(split_x, adjoint=adjoint, adjoint_arg=adjoint_arg)
            self.assertEqual(len(split_matmul), len(operator.operators))
            split_matmul = linear_operator_util.broadcast_matrix_batch_dims(split_matmul)
            fused_block_matmul = array_ops.concat(split_matmul, axis=-2)
            op_matmul_v, mat_matmul_v, fused_block_matmul_v = sess.run([op_matmul, mat_matmul, fused_block_matmul])
            self.assertAC(fused_block_matmul_v, mat_matmul_v)
        else:
            op_matmul_v, mat_matmul_v = sess.run([op_matmul, mat_matmul])
        self.assertAC(op_matmul_v, mat_matmul_v)