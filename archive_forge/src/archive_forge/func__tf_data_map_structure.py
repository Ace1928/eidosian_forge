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
def _tf_data_map_structure(func, *structure, **check_types_dict):
    if not callable(func):
        raise TypeError(f'Argument `func` must be callable, got: {func}')
    if not structure:
        raise ValueError('Must provide at least one structure')
    if check_types_dict:
        if 'check_types' not in check_types_dict or len(check_types_dict) > 1:
            raise ValueError(f"Only valid keyword argument for `check_types_dict` is 'check_types'. Got {check_types_dict}.")
        check_types = check_types_dict['check_types']
    else:
        check_types = True
    for other in structure[1:]:
        _tf_data_assert_same_structure(structure[0], other, check_types=check_types)
    flat_structure = (_tf_data_flatten(s) for s in structure)
    entries = zip(*flat_structure)
    return _tf_data_pack_sequence_as(structure[0], [func(*x) for x in entries])