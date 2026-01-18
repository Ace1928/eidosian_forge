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
Returns a list of independent `Generator` objects.

    Two generators are independent of each other in the sense that the
    random-number streams they generate don't have statistically detectable
    correlations. The new generators are also independent of the old one.
    The old generator's state will be changed (like other random-number
    generating methods), so two calls of `split` will return different
    new generators.

    For example:

    ```python
    gens = get_global_generator().split(count=10)
    for gen in gens:
      numbers = gen.normal(shape=[2, 3])
      # ...
    gens2 = get_global_generator().split(count=10)
    # gens2 will be different from gens
    ```

    The new generators will be put on the current device (possible different
    from the old generator's), for example:

    ```python
    with tf.device("/device:CPU:0"):
      gen = Generator(seed=1234)  # gen is on CPU
    with tf.device("/device:GPU:0"):
      gens = gen.split(count=10)  # gens are on GPU
    ```

    Args:
      count: the number of generators to return.

    Returns:
      A list (length `count`) of `Generator` objects independent of each other.
      The new generators have the same RNG algorithm as the old one.
    