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
class _Mapping(collections.namedtuple('_Mapping', ['x', 'y', 'ildj_map', 'kwargs'])):
    """Helper class to make it easier to manage caching in `Bijector`."""

    def __new__(cls, x=None, y=None, ildj_map=None, kwargs=None):
        """Custom __new__ so namedtuple items have defaults.

    Args:
      x: `Tensor`. Forward.
      y: `Tensor`. Inverse.
      ildj_map: `Dictionary`. This is a mapping from event_ndims to a `Tensor`
        representing the inverse log det jacobian.
      kwargs: Python dictionary. Extra args supplied to
        forward/inverse/etc functions.

    Returns:
      mapping: New instance of _Mapping.
    """
        return super(_Mapping, cls).__new__(cls, x, y, ildj_map, kwargs)

    @property
    def x_key(self):
        """Returns key used for caching Y=g(X)."""
        return (object_identity.Reference(self.x),) + self._deep_tuple(tuple(sorted(self.kwargs.items())))

    @property
    def y_key(self):
        """Returns key used for caching X=g^{-1}(Y)."""
        return (object_identity.Reference(self.y),) + self._deep_tuple(tuple(sorted(self.kwargs.items())))

    def merge(self, x=None, y=None, ildj_map=None, kwargs=None, mapping=None):
        """Returns new _Mapping with args merged with self.

    Args:
      x: `Tensor`. Forward.
      y: `Tensor`. Inverse.
      ildj_map: `Dictionary`. This is a mapping from event_ndims to a `Tensor`
        representing the inverse log det jacobian.
      kwargs: Python dictionary. Extra args supplied to
        forward/inverse/etc functions.
      mapping: Instance of _Mapping to merge. Can only be specified if no other
        arg is specified.

    Returns:
      mapping: New instance of `_Mapping` which has inputs merged with self.

    Raises:
      ValueError: if mapping and any other arg is not `None`.
    """
        if mapping is None:
            mapping = _Mapping(x=x, y=y, ildj_map=ildj_map, kwargs=kwargs)
        elif any((arg is not None for arg in [x, y, ildj_map, kwargs])):
            raise ValueError('Cannot simultaneously specify mapping and individual arguments.')
        return _Mapping(x=self._merge(self.x, mapping.x), y=self._merge(self.y, mapping.y), ildj_map=self._merge_dicts(self.ildj_map, mapping.ildj_map), kwargs=self._merge(self.kwargs, mapping.kwargs))

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

    def _merge(self, old, new):
        """Helper to merge which handles merging one value."""
        if old is None:
            return new
        elif new is not None and old is not new:
            raise ValueError('Incompatible values: %s != %s' % (old, new))
        return old

    def _deep_tuple(self, x):
        """Converts lists of lists to tuples of tuples."""
        return tuple(map(self._deep_tuple, x)) if isinstance(x, (list, tuple)) else x