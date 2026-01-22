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
def _tf_data_sorted(dict_):
    """Returns a sorted list of the dict keys, with error if keys not sortable."""
    try:
        return sorted(list(dict_))
    except TypeError as e:
        raise TypeError(f'nest only supports dicts with sortable keys. Error: {e.message}')