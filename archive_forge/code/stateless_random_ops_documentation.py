import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_random_index_shuffle_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import shape_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
Outputs random values from a truncated normal distribution.

  The generated values follow a normal distribution with specified mean and
  standard deviation, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.


  Examples:

  Sample from a Truncated normal, with deferring shape parameters that
  broadcast.

  >>> means = 0.
  >>> stddevs = tf.math.exp(tf.random.uniform(shape=[2, 3]))
  >>> minvals = [-1., -2., -1000.]
  >>> maxvals = [[10000.], [1.]]
  >>> y = tf.random.stateless_parameterized_truncated_normal(
  ...   shape=[10, 2, 3], seed=[7, 17],
  ...   means=means, stddevs=stddevs, minvals=minvals, maxvals=maxvals)
  >>> y.shape
  TensorShape([10, 2, 3])

  Args:
    shape: A 1-D integer `Tensor` or Python array. The shape of the output
      tensor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    means: A `Tensor` or Python value of type `dtype`. The mean of the truncated
      normal distribution. This must broadcast with `stddevs`, `minvals` and
      `maxvals`, and the broadcasted shape must be dominated by `shape`.
    stddevs: A `Tensor` or Python value of type `dtype`. The standard deviation
      of the truncated normal distribution. This must broadcast with `means`,
      `minvals` and `maxvals`, and the broadcasted shape must be dominated by
      `shape`.
    minvals: A `Tensor` or Python value of type `dtype`. The minimum value of
      the truncated normal distribution. This must broadcast with `means`,
      `stddevs` and `maxvals`, and the broadcasted shape must be dominated by
      `shape`.
    maxvals: A `Tensor` or Python value of type `dtype`. The maximum value of
      the truncated normal distribution. This must broadcast with `means`,
      `stddevs` and `minvals`, and the broadcasted shape must be dominated by
      `shape`.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random truncated normal values.
  