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
Calculate math_ops.cumsum for a RaggedTensor.

  Given a ragged tensor `x`, the `result` is a ragged tensor with the same
  shape. One can calculate the value of `result[i_1...i_k]` as follows:
  ```
  dense_result=tf.math.cumsum(rt.to_tensor(), axis=axis, exclusive=exclusive,
                              reverse=reverse)
  result[i_1...i_k]=dense_result[i_1...i_k]
  ```

  Args:
    x: the original ragged tensor to sum.
    axis: the axis along which to sum, can range -rank<=axis<rank.
    exclusive: is the sum exclusive or inclusive? If True, then result[0]=0.
        If False, then result[0]=x[0].
    reverse: If True, sum from back to front.
    name: the name of the op.
  Returns:
    the cumulative sum.
  