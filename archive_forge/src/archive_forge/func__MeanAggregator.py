import functools
import itertools
import operator
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
def _MeanAggregator(inputs, segments):
    """Replaces each segment with its mean along the last axis.

  Specifically, each value in the `inputs` tensor gets replaced by the mean
  value computed from the values that belong to the same segment.

  Args:
   inputs: A 2-tensor. Aggregation is done over dimension 1.
   segments: A 2-tensor, same shape as `input`.

  Returns:
    The result, same shape and type as `inputs`.
  """
    result = []
    for inputs_i, segments_i in zip(array_ops.split(inputs, inputs.shape[0]), array_ops.split(segments, segments.shape[0])):
        means_i = math_ops.unsorted_segment_mean(inputs_i, segments_i, num_segments=math_ops.reduce_max(segments_i) + 1)
        result.append(array_ops.reshape(array_ops.gather(means_i, segments_i), [-1]))
    return array_ops_stack.stack(result, axis=0)