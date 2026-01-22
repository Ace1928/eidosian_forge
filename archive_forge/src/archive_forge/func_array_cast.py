import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
@_wrap_for_chunked_arrays
def array_cast(array: pa.Array, pa_type: pa.DataType, allow_number_to_str=True):
    """Improved version of `pa.Array.cast`

    It supports casting `pa.StructArray` objects to re-order the fields.
    It also let you control certain aspects of the casting, e.g. whether
    to disable numbers (`floats` or `ints`) to strings.

    Args:
        array (`pa.Array`):
            PyArrow array to cast
        pa_type (`pa.DataType`):
            Target PyArrow type
        allow_number_to_str (`bool`, defaults to `True`):
            Whether to allow casting numbers to strings.
            Defaults to `True`.

    Raises:
        `pa.ArrowInvalidError`: if the arrow data casting fails
        `TypeError`: if the target type is not supported according, e.g.

            - if a field is missing
            - if casting from numbers to strings and `allow_number_to_str` is `False`

    Returns:
        `List[pyarrow.Array]`: the casted array
    """
    _c = partial(array_cast, allow_number_to_str=allow_number_to_str)
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if isinstance(pa_type, pa.ExtensionType):
        return pa_type.wrap_array(_c(array, pa_type.storage_type))
    elif array.type == pa_type:
        return array
    elif pa.types.is_struct(array.type):
        if pa.types.is_struct(pa_type) and {field.name for field in pa_type} == {field.name for field in array.type}:
            if array.type.num_fields == 0:
                return array
            arrays = [_c(array.field(field.name), field.type) for field in pa_type]
            return pa.StructArray.from_arrays(arrays, fields=list(pa_type), mask=array.is_null())
    elif pa.types.is_list(array.type):
        if pa.types.is_fixed_size_list(pa_type):
            if _are_list_values_of_length(array, pa_type.list_size):
                if array.null_count > 0:
                    array_type = array.type
                    storage_type = _storage_type(array_type)
                    if array_type != storage_type:
                        array = _c(array, storage_type)
                        array = pc.list_slice(array, 0, pa_type.list_size, return_fixed_size_list=True)
                        array = _c(array, array_type)
                    else:
                        array = pc.list_slice(array, 0, pa_type.list_size, return_fixed_size_list=True)
                    array_values = array.values
                    if config.PYARROW_VERSION.major < 15:
                        return pa.Array.from_buffers(pa_type, len(array), [array.is_valid().buffers()[1]], children=[_c(array_values, pa_type.value_type)])
                    else:
                        return pa.FixedSizeListArray.from_arrays(_c(array_values, pa_type.value_type), pa_type.list_size, mask=array.is_null())
                else:
                    array_values = array.values[array.offset * pa_type.length:(array.offset + len(array)) * pa_type.length]
                    return pa.FixedSizeListArray.from_arrays(_c(array_values, pa_type.value_type), pa_type.list_size)
        elif pa.types.is_list(pa_type):
            array_offsets = _combine_list_array_offsets_with_mask(array)
            return pa.ListArray.from_arrays(array_offsets, _c(array.values, pa_type.value_type))
    elif pa.types.is_fixed_size_list(array.type):
        if pa.types.is_fixed_size_list(pa_type):
            if pa_type.list_size == array.type.list_size:
                array_values = array.values[array.offset * array.type.list_size:(array.offset + len(array)) * array.type.list_size]
                if config.PYARROW_VERSION.major < 15:
                    return pa.Array.from_buffers(pa_type, len(array), [array.is_valid().buffers()[1]], children=[_c(array_values, pa_type.value_type)])
                else:
                    return pa.FixedSizeListArray.from_arrays(_c(array_values, pa_type.value_type), pa_type.list_size, mask=array.is_null())
        elif pa.types.is_list(pa_type):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.ListArray.from_arrays(array_offsets, _c(array.values, pa_type.value_type), mask=array.is_null())
    else:
        if not allow_number_to_str and pa.types.is_string(pa_type) and (pa.types.is_floating(array.type) or pa.types.is_integer(array.type)):
            raise TypeError(f"Couldn't cast array of type {array.type} to {pa_type} since allow_number_to_str is set to {allow_number_to_str}")
        if pa.types.is_null(pa_type) and (not pa.types.is_null(array.type)):
            raise TypeError(f"Couldn't cast array of type {array.type} to {pa_type}")
        return array.cast(pa_type)
    raise TypeError(f"Couldn't cast array of type\n{array.type}\nto\n{pa_type}")