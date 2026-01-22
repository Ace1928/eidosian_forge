import math
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import special_math
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class LaplaceWithSoftplusScale(Laplace):
    """Laplace with softplus applied to `scale`."""

    @deprecation.deprecated('2019-01-01', 'Use `tfd.Laplace(loc, tf.nn.softplus(scale)) instead.', warn_once=True)
    def __init__(self, loc, scale, validate_args=False, allow_nan_stats=True, name='LaplaceWithSoftplusScale'):
        parameters = dict(locals())
        with ops.name_scope(name, values=[loc, scale]) as name:
            super(LaplaceWithSoftplusScale, self).__init__(loc=loc, scale=nn.softplus(scale, name='softplus_scale'), validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name)
        self._parameters = parameters