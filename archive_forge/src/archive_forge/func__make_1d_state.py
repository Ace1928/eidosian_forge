from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _make_1d_state(state_size, seed):
    """Makes a 1-D RNG state.

  Args:
    state_size: an integer.
    seed: an integer or 1-D tensor.

  Returns:
    a 1-D tensor of shape [state_size] and dtype STATE_TYPE.
  """
    if isinstance(seed, int):
        ls = []
        for _ in range(state_size):
            ls.append(seed & SEED_BIT_MASK)
            seed >>= SEED_TYPE_BITS
        seed = ls
    seed = nest.map_structure(_uint_to_int, seed)
    seed = math_ops.cast(seed, STATE_TYPE)
    seed = array_ops.reshape(seed, [-1])
    seed = seed[0:state_size]
    seed_size = seed.shape[0]
    if seed_size is None:
        seed_size = array_ops.shape(seed)[0]
    padding_size = math_ops.maximum(state_size - seed_size, 0)
    padding = array_ops.zeros([padding_size], seed.dtype)
    seed = array_ops.concat([padding, seed], axis=0)
    seed.set_shape([state_size])
    return seed