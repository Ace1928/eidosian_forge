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
def _call_forward(self, x, name, **kwargs):
    with self._name_scope(name, [x]):
        x = ops.convert_to_tensor(x, name='x')
        self._maybe_assert_dtype(x)
        if not self._is_injective:
            return self._forward(x, **kwargs)
        mapping = self._lookup(x=x, kwargs=kwargs)
        if mapping.y is not None:
            return mapping.y
        mapping = mapping.merge(y=self._forward(x, **kwargs))
        self._cache(mapping)
        return mapping.y