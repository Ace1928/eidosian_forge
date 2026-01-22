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
def _tf_data_pack_sequence_as(structure, flat_sequence):
    """Returns a given flattened sequence packed into a nest.

  If `structure` is a scalar, `flat_sequence` must be a single-element list;
  in this case the return value is `flat_sequence[0]`.

  Args:
    structure: tuple or list constructed of scalars and/or other tuples/lists,
      or a scalar.  Note: numpy arrays are considered scalars.
    flat_sequence: flat sequence to pack.

  Returns:
    packed: `flat_sequence` converted to have the same recursive structure as
      `structure`.

  Raises:
    ValueError: If nest and structure have different element counts.
  """
    if not (_tf_data_is_nested(flat_sequence) or isinstance(flat_sequence, list)):
        raise TypeError(f"Argument `flat_sequence` must be a sequence. Got '{type(flat_sequence).__name__}'.")
    if not _tf_data_is_nested(structure):
        if len(flat_sequence) != 1:
            raise ValueError(f'Argument `structure` is a scalar but `len(flat_sequence)`={len(flat_sequence)} > 1')
        return flat_sequence[0]
    flat_structure = _tf_data_flatten(structure)
    if len(flat_structure) != len(flat_sequence):
        raise ValueError(f'Could not pack sequence. Argument `structure` had {len(flat_structure)} elements, but argument `flat_sequence` had {len(flat_sequence)} elements. Received structure: {structure}, flat_sequence: {flat_sequence}.')
    _, packed = _tf_data_packed_nest_with_indices(structure, flat_sequence, 0)
    return sequence_like(structure, packed)