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
def _tf_data_flatten_up_to(shallow_tree, input_tree):
    _tf_data_assert_shallow_structure(shallow_tree, input_tree)
    return list(_tf_data_yield_flat_up_to(shallow_tree, input_tree))