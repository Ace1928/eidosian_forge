import enum
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
Determines the key and counter for Philox PRNG with the given seed.

  Args:
    seed: An integer tensor of shape [2]. The seed to calculate the key and
      counter from.

  Returns:
    A pair (key, counter) suitable for V2 stateless RNG ops like
    `StatelessRandomUniformV2`.
  