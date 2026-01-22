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
def _merge_dicts(self, old=None, new=None):
    """Helper to merge two dictionaries."""
    old = {} if old is None else old
    new = {} if new is None else new
    for k, v in new.items():
        val = old.get(k, None)
        if val is not None and val is not v:
            raise ValueError('Found different value for existing key (key:{} old_value:{} new_value:{}'.format(k, old[k], v))
        old[k] = v
    return old