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
def yield_value(modality, iterable):
    """Yield elements of `iterable` in a deterministic order.

  Args:
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    iterable: an iterable.

  Yields:
    The iterable elements in a deterministic order.
  """
    if modality == Modality.CORE:
        yield from _tf_core_yield_value(iterable)
    elif modality == Modality.DATA:
        yield from _tf_data_yield_value(iterable)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))