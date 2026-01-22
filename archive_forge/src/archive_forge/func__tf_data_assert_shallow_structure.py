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
def _tf_data_assert_shallow_structure(shallow_tree, input_tree, check_types=True):
    if _tf_data_is_nested(shallow_tree):
        if not _tf_data_is_nested(input_tree):
            raise TypeError(f"If shallow structure is a sequence, input must also be a sequence. Input has type: '{type(input_tree).__name__}'.")
        if check_types and (not isinstance(input_tree, type(shallow_tree))):
            raise TypeError(f"The two structures don't have the same sequence type. Input structure has type '{type(input_tree).__name__}', while shallow structure has type '{type(shallow_tree).__name__}'.")
        if len(input_tree) != len(shallow_tree):
            raise ValueError(f"The two structures don't have the same sequence length. Input structure has length {len(input_tree)}, while shallow structure has length {len(shallow_tree)}.")
        if check_types and isinstance(shallow_tree, _collections_abc.Mapping):
            if set(input_tree) != set(shallow_tree):
                raise ValueError(f"The two structures don't have the same keys. Input structure has keys {list(input_tree)}, while shallow structure has keys {list(shallow_tree)}.")
            input_tree = sorted(input_tree.items())
            shallow_tree = sorted(shallow_tree.items())
        for shallow_branch, input_branch in zip(shallow_tree, input_tree):
            _tf_data_assert_shallow_structure(shallow_branch, input_branch, check_types=check_types)