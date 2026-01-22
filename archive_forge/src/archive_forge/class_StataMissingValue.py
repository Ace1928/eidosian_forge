from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
class StataMissingValue:
    """
    An observation's missing value.

    Parameters
    ----------
    value : {int, float}
        The Stata missing value code

    Notes
    -----
    More information: <https://www.stata.com/help.cgi?missing>

    Integer missing values make the code '.', '.a', ..., '.z' to the ranges
    101 ... 127 (for int8), 32741 ... 32767  (for int16) and 2147483621 ...
    2147483647 (for int32).  Missing values for floating point data types are
    more complex but the pattern is simple to discern from the following table.

    np.float32 missing values (float in Stata)
    0000007f    .
    0008007f    .a
    0010007f    .b
    ...
    00c0007f    .x
    00c8007f    .y
    00d0007f    .z

    np.float64 missing values (double in Stata)
    000000000000e07f    .
    000000000001e07f    .a
    000000000002e07f    .b
    ...
    000000000018e07f    .x
    000000000019e07f    .y
    00000000001ae07f    .z
    """
    MISSING_VALUES: dict[float, str] = {}
    bases: Final = (101, 32741, 2147483621)
    for b in bases:
        MISSING_VALUES[b] = '.'
        for i in range(1, 27):
            MISSING_VALUES[i + b] = '.' + chr(96 + i)
    float32_base: bytes = b'\x00\x00\x00\x7f'
    increment_32: int = struct.unpack('<i', b'\x00\x08\x00\x00')[0]
    for i in range(27):
        key = struct.unpack('<f', float32_base)[0]
        MISSING_VALUES[key] = '.'
        if i > 0:
            MISSING_VALUES[key] += chr(96 + i)
        int_value = struct.unpack('<i', struct.pack('<f', key))[0] + increment_32
        float32_base = struct.pack('<i', int_value)
    float64_base: bytes = b'\x00\x00\x00\x00\x00\x00\xe0\x7f'
    increment_64 = struct.unpack('q', b'\x00\x00\x00\x00\x00\x01\x00\x00')[0]
    for i in range(27):
        key = struct.unpack('<d', float64_base)[0]
        MISSING_VALUES[key] = '.'
        if i > 0:
            MISSING_VALUES[key] += chr(96 + i)
        int_value = struct.unpack('q', struct.pack('<d', key))[0] + increment_64
        float64_base = struct.pack('q', int_value)
    BASE_MISSING_VALUES: Final = {'int8': 101, 'int16': 32741, 'int32': 2147483621, 'float32': struct.unpack('<f', float32_base)[0], 'float64': struct.unpack('<d', float64_base)[0]}

    def __init__(self, value: float) -> None:
        self._value = value
        value = int(value) if value < 2147483648 else float(value)
        self._str = self.MISSING_VALUES[value]

    @property
    def string(self) -> str:
        """
        The Stata representation of the missing value: '.', '.a'..'.z'

        Returns
        -------
        str
            The representation of the missing value.
        """
        return self._str

    @property
    def value(self) -> float:
        """
        The binary representation of the missing value.

        Returns
        -------
        {int, float}
            The binary representation of the missing value.
        """
        return self._value

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f'{type(self)}({self})'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.string == other.string and (self.value == other.value)

    @classmethod
    def get_base_missing_value(cls, dtype: np.dtype) -> float:
        if dtype.type is np.int8:
            value = cls.BASE_MISSING_VALUES['int8']
        elif dtype.type is np.int16:
            value = cls.BASE_MISSING_VALUES['int16']
        elif dtype.type is np.int32:
            value = cls.BASE_MISSING_VALUES['int32']
        elif dtype.type is np.float32:
            value = cls.BASE_MISSING_VALUES['float32']
        elif dtype.type is np.float64:
            value = cls.BASE_MISSING_VALUES['float64']
        else:
            raise ValueError('Unsupported dtype')
        return value