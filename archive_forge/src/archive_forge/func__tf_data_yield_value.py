import collections as _collections
import enum
import typing
from typing import Protocol
import six as _six
import wrapt as _wrapt
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.compat import collections_abc as _collections_abc
def _tf_data_yield_value(iterable):
    """Yield elements of `iterable` in a deterministic order.

  Args:
    iterable: an iterable.

  Yields:
    The iterable elements in a deterministic order.
  """
    if isinstance(iterable, _collections_abc.Mapping):
        for key in _tf_data_sorted(iterable):
            yield iterable[key]
    elif iterable.__class__.__name__ == 'SparseTensorValue':
        yield iterable
    elif _is_attrs(iterable):
        for _, attr in _get_attrs_items(iterable):
            yield attr
    elif isinstance(iterable, CustomNestProtocol):
        flat_component = iterable.__tf_flatten__()[1]
        assert isinstance(flat_component, tuple)
        yield from flat_component
    else:
        for value in iterable:
            yield value