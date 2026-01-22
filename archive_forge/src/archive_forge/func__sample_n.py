from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _sample_n(self, n, seed=None):
    n_draws = math_ops.cast(self.total_count, dtype=dtypes.int32)
    k = self.event_shape_tensor()[0]
    unnormalized_logits = array_ops.reshape(math_ops.log(random_ops.random_gamma(shape=[n], alpha=self.concentration, dtype=self.dtype, seed=seed)), shape=[-1, k])
    draws = random_ops.multinomial(logits=unnormalized_logits, num_samples=n_draws, seed=distribution_util.gen_new_seed(seed, salt='dirichlet_multinomial'))
    x = math_ops.reduce_sum(array_ops.one_hot(draws, depth=k), -2)
    final_shape = array_ops.concat([[n], self.batch_shape_tensor(), [k]], 0)
    x = array_ops.reshape(x, final_shape)
    return math_ops.cast(x, self.dtype)