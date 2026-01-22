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
def _merge(self, old, new):
    """Helper to merge which handles merging one value."""
    if old is None:
        return new
    elif new is not None and old is not new:
        raise ValueError('Incompatible values: %s != %s' % (old, new))
    return old