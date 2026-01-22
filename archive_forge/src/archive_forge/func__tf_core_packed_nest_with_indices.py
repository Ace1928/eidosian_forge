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
def _tf_core_packed_nest_with_indices(structure, flat, index, is_nested_fn, sequence_fn=None):
    """Helper function for pack_sequence_as.

  Args:
    structure: structure to mimic.
    flat: Flattened values to output substructure for.
    index: Index at which to start reading from flat.
    is_nested_fn: Function used to test if a value should be treated as a nested
      structure.
    sequence_fn: Function used to generate a new strcuture instance.

  Returns:
    The tuple (new_index, child), where:
      * new_index - the updated index into `flat` having processed `structure`.
      * packed - the subset of `flat` corresponding to `structure`,
                 having started at `index`, and packed into the same nested
                 format.

  Raises:
    ValueError: if `structure` contains more atoms than `flat`
      (assuming indexing starts from `index`).
  """
    packed = []
    sequence_fn = sequence_fn or sequence_like
    for s in _tf_core_yield_value(structure):
        if is_nested_fn(s):
            new_index, child = _tf_core_packed_nest_with_indices(s, flat, index, is_nested_fn, sequence_fn)
            packed.append(sequence_fn(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return (index, packed)