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
def yield_sorted_items(modality, iterable):
    if modality == Modality.CORE:
        return _tf_core_yield_sorted_items(iterable)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))