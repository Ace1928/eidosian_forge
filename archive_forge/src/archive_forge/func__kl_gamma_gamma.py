import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@kullback_leibler.RegisterKL(Gamma, Gamma)
def _kl_gamma_gamma(g0, g1, name=None):
    """Calculate the batched KL divergence KL(g0 || g1) with g0 and g1 Gamma.

  Args:
    g0: instance of a Gamma distribution object.
    g1: instance of a Gamma distribution object.
    name: (optional) Name to use for created operations.
      Default is "kl_gamma_gamma".

  Returns:
    kl_gamma_gamma: `Tensor`. The batchwise KL(g0 || g1).
  """
    with ops.name_scope(name, 'kl_gamma_gamma', values=[g0.concentration, g0.rate, g1.concentration, g1.rate]):
        return (g0.concentration - g1.concentration) * math_ops.digamma(g0.concentration) + math_ops.lgamma(g1.concentration) - math_ops.lgamma(g0.concentration) + g1.concentration * math_ops.log(g0.rate) - g1.concentration * math_ops.log(g1.rate) + g0.concentration * (g1.rate / g0.rate - 1.0)