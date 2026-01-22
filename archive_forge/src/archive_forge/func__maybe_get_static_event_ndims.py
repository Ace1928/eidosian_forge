import abc
import collections
import contextlib
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import object_identity
def _maybe_get_static_event_ndims(self, event_ndims):
    """Helper which returns tries to return an integer static value."""
    event_ndims_ = distribution_util.maybe_get_static_value(event_ndims)
    if isinstance(event_ndims_, (np.generic, np.ndarray)):
        if event_ndims_.dtype not in (np.int32, np.int64):
            raise ValueError('Expected integer dtype, got dtype {}'.format(event_ndims_.dtype))
        if isinstance(event_ndims_, np.ndarray) and len(event_ndims_.shape):
            raise ValueError('Expected a scalar integer, got {}'.format(event_ndims_))
        event_ndims_ = int(event_ndims_)
    return event_ndims_