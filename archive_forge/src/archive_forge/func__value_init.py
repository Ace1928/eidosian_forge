import typing
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from functools import lru_cache
from numbers import Number
from types import MappingProxyType
from typing import (
from pennylane.data.base import hdf5
from pennylane.data.base.hdf5 import HDF5, HDF5Any, HDF5Group
from pennylane.data.base.typing_util import UNSET, get_type, get_type_str
def _value_init(self, value: Union[InitValueType, Literal[UNSET]], info: Optional[AttributeInfo], parent_and_key: Optional[Tuple[HDF5Group, str]]):
    """Constructor for value initialization. See __init__()."""
    if parent_and_key is not None:
        parent, key = parent_and_key
    else:
        parent, key = (hdf5.create_group(), '_')
    if value is UNSET:
        value = self.default_value()
        if value is UNSET:
            raise TypeError("__init__() missing 1 required positional argument: 'value'")
    self._bind = self._set_value(value, info, parent, key)
    self._check_bind()
    self.__post_init__(value)