from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _sample_single(args):
    logits, n_draw = (args[0], args[1])
    x = random_ops.multinomial(logits[array_ops.newaxis, ...], n_draw, seed)
    x = array_ops.reshape(x, shape=[n, -1])
    x = math_ops.reduce_sum(array_ops.one_hot(x, depth=k), axis=-2)
    return x