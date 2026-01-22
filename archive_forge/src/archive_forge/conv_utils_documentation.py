import itertools
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
Returns `unsqueeze_batch(op(squeeze_batch(inp)))`.

  Where `squeeze_batch` reshapes `inp` to shape
  `[prod(inp.shape[:-inner_rank])] + inp.shape[-inner_rank:]`
  and `unsqueeze_batch` does the reverse reshape but on the output.

  Args:
    inp: A tensor with dims `batch_shape + inner_shape` where `inner_shape`
      is length `inner_rank`.
    op: A callable that takes a single input tensor and returns a single.
      output tensor.
    inner_rank: A python integer.

  Returns:
    `unsqueeze_batch_op(squeeze_batch(inp))`.
  