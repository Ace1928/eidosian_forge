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
class BetaWithSoftplusConcentration(Beta):
    """Beta with softplus transform of `concentration1` and `concentration0`."""

    @deprecation.deprecated('2019-01-01', 'Use `tfd.Beta(tf.nn.softplus(concentration1), tf.nn.softplus(concentration2))` instead.', warn_once=True)
    def __init__(self, concentration1, concentration0, validate_args=False, allow_nan_stats=True, name='BetaWithSoftplusConcentration'):
        parameters = dict(locals())
        with ops.name_scope(name, values=[concentration1, concentration0]) as name:
            super(BetaWithSoftplusConcentration, self).__init__(concentration1=nn.softplus(concentration1, name='softplus_concentration1'), concentration0=nn.softplus(concentration0, name='softplus_concentration0'), validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name)
        self._parameters = parameters