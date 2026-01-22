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
class GammaWithSoftplusConcentrationRate(Gamma):
    """`Gamma` with softplus of `concentration` and `rate`."""

    @deprecation.deprecated('2019-01-01', 'Use `tfd.Gamma(tf.nn.softplus(concentration), tf.nn.softplus(rate))` instead.', warn_once=True)
    def __init__(self, concentration, rate, validate_args=False, allow_nan_stats=True, name='GammaWithSoftplusConcentrationRate'):
        parameters = dict(locals())
        with ops.name_scope(name, values=[concentration, rate]) as name:
            super(GammaWithSoftplusConcentrationRate, self).__init__(concentration=nn.softplus(concentration, name='softplus_concentration'), rate=nn.softplus(rate, name='softplus_rate'), validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name)
        self._parameters = parameters