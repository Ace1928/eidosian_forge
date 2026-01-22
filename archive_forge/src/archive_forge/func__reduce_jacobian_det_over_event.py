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
def _reduce_jacobian_det_over_event(self, y, ildj, min_event_ndims, event_ndims):
    """Reduce jacobian over event_ndims - min_event_ndims."""
    y_rank = array_ops.rank(y)
    y_shape = array_ops.shape(y)[y_rank - event_ndims:y_rank - min_event_ndims]
    ones = array_ops.ones(y_shape, ildj.dtype)
    reduced_ildj = math_ops.reduce_sum(ones * ildj, axis=self._get_event_reduce_dims(min_event_ndims, event_ndims))
    event_ndims_ = self._maybe_get_static_event_ndims(event_ndims)
    if event_ndims_ is not None and y.shape.ndims is not None and (ildj.shape.ndims is not None):
        y_shape = y.shape[y.shape.ndims - event_ndims_:y.shape.ndims - min_event_ndims]
        broadcast_shape = array_ops.broadcast_static_shape(ildj.shape, y_shape)
        reduced_ildj.set_shape(broadcast_shape[:broadcast_shape.ndims - (event_ndims_ - min_event_ndims)])
    return reduced_ildj