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
def _storage_type(type: pa.DataType) -> pa.DataType:
    """Convert a (possibly nested) `pa.ExtensionType` to its storage type."""
    if isinstance(type, pa.ExtensionType):
        return _storage_type(type.storage_type)
    elif isinstance(type, pa.StructType):
        return pa.struct([pa.field(field.name, _storage_type(field.type)) for field in type])
    elif isinstance(type, pa.ListType):
        return pa.list_(_storage_type(type.value_type))
    elif isinstance(type, pa.FixedSizeListType):
        return pa.list_(_storage_type(type.value_type), type.list_size)
    return type