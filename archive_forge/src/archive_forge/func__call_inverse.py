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
def _call_inverse(self, y, name, **kwargs):
    with self._name_scope(name, [y]):
        y = ops.convert_to_tensor(y, name='y')
        self._maybe_assert_dtype(y)
        if not self._is_injective:
            return self._inverse(y, **kwargs)
        mapping = self._lookup(y=y, kwargs=kwargs)
        if mapping.x is not None:
            return mapping.x
        mapping = mapping.merge(x=self._inverse(y, **kwargs))
        self._cache(mapping)
        return mapping.x