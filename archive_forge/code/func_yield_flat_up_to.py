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
def yield_flat_up_to(modality, shallow_tree, input_tree, is_nested_fn, path=()):
    """Yields (path, value) pairs of input_tree flattened up to shallow_tree.

  - For Modality.CORE: See comments for _tf_core_yield_flat_up_to() below
  - For Modality.DATA: See comments for _tf_data_yield_flat_up_to() below

  Args:
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    shallow_tree: Nested structure. Traverse no further than its leaf nodes.
    input_tree: Nested structure. Return the paths and values from this tree.
      Must have the same upper structure as shallow_tree.
    is_nested_fn: Arg valid for Modality.CORE only. Function used to test if a
      value should be treated as a nested structure.
    path: Arg valid for Modality.CORE only. Tuple. Optional argument, only used
      when recursing. The path from the root of the original shallow_tree, down
      to the root of the shallow_tree arg of this recursive call.

  Yields:
    Pairs of (path, value), where path the tuple path of a leaf node in
    shallow_tree, and value is the value of the corresponding node in
    input_tree.
  """
    if modality == Modality.CORE:
        yield from _tf_core_yield_flat_up_to(shallow_tree, input_tree, is_nested_fn, path)
    elif modality == Modality.DATA:
        yield from _tf_data_yield_flat_up_to(shallow_tree, input_tree)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))