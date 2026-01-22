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
def _get_event_reduce_dims(self, min_event_ndims, event_ndims):
    """Compute the reduction dimensions given event_ndims."""
    event_ndims_ = self._maybe_get_static_event_ndims(event_ndims)
    if event_ndims_ is not None:
        return [-index for index in range(1, event_ndims_ - min_event_ndims + 1)]
    else:
        reduce_ndims = event_ndims - min_event_ndims
        return math_ops.range(-reduce_ndims, 0)