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
def test_slicing(self):
    with self.session(graph=ops.Graph()) as sess:
        sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
        operator, mat = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
        batch_shape = shapes_info.shape[:-2]
        if not batch_shape or batch_shape[0] <= 1:
            return
        slices = [slice(1, -1)]
        if len(batch_shape) > 1:
            slices += [..., slice(0, 1)]
        sliced_operator = operator[slices]
        matrix_slices = slices + [slice(None), slice(None)]
        sliced_matrix = mat[matrix_slices]
        sliced_op_dense = sliced_operator.to_dense()
        op_dense_v, mat_v = sess.run([sliced_op_dense, sliced_matrix])
        self.assertAC(op_dense_v, mat_v)