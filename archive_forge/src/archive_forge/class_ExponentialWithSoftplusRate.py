import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import gamma
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class ExponentialWithSoftplusRate(Exponential):
    """Exponential with softplus transform on `rate`."""

    @deprecation.deprecated('2019-01-01', 'Use `tfd.Exponential(tf.nn.softplus(rate)).', warn_once=True)
    def __init__(self, rate, validate_args=False, allow_nan_stats=True, name='ExponentialWithSoftplusRate'):
        parameters = dict(locals())
        with ops.name_scope(name, values=[rate]) as name:
            super(ExponentialWithSoftplusRate, self).__init__(rate=nn.softplus(rate, name='softplus_rate'), validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name)
        self._parameters = parameters