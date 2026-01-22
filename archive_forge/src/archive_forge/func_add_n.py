import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@dispatch.dispatch_for_api(math_ops.add_n)
def add_n(inputs: typing.List[ragged_tensor.RaggedOrDense], name=None):
    """RaggedTensor implementation for tf.math.add_n."""
    if len(inputs) < 0:
        raise ValueError('tf.add_n: expected at least one input.')
    with ops.name_scope(name, 'RaggedAddN', inputs):
        return ragged_functional_ops.map_flat_values(math_ops.add_n, inputs)