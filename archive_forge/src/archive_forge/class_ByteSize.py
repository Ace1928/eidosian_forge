from __future__ import annotations as _annotations
import base64
import dataclasses as _dataclasses
import re
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import (
from uuid import UUID
import annotated_types
from annotated_types import BaseMetadata, MaxLen, MinLen
from pydantic_core import CoreSchema, PydanticCustomError, core_schema
from typing_extensions import Annotated, Literal, Protocol, TypeAlias, TypeAliasType, deprecated
from ._internal import (
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .errors import PydanticUserError
from .json_schema import JsonSchemaValue
from .warnings import PydanticDeprecatedSince20
from pydantic import BaseModel, PositiveInt, ValidationError
from pydantic import BaseModel, NegativeInt, ValidationError
from pydantic import BaseModel, NonPositiveInt, ValidationError
from pydantic import BaseModel, NonNegativeInt, ValidationError
from pydantic import BaseModel, StrictInt, ValidationError
from pydantic import BaseModel, PositiveFloat, ValidationError
from pydantic import BaseModel, NegativeFloat, ValidationError
from pydantic import BaseModel, NonPositiveFloat, ValidationError
from pydantic import BaseModel, NonNegativeFloat, ValidationError
from pydantic import BaseModel, StrictFloat, ValidationError
from pydantic import BaseModel, FiniteFloat
import uuid
from pydantic import UUID1, BaseModel
import uuid
from pydantic import UUID3, BaseModel
import uuid
from pydantic import UUID4, BaseModel
import uuid
from pydantic import UUID5, BaseModel
from pathlib import Path
from pydantic import BaseModel, FilePath, ValidationError
from pathlib import Path
from pydantic import BaseModel, DirectoryPath, ValidationError
from pydantic import Base64Bytes, BaseModel, ValidationError
from pydantic import Base64Str, BaseModel, ValidationError
from pydantic import Base64UrlBytes, BaseModel
from pydantic import Base64UrlStr, BaseModel
class ByteSize(int):
    """Converts a string representing a number of bytes with units (such as `'1KB'` or `'11.5MiB'`) into an integer.

    You can use the `ByteSize` data type to (case-insensitively) convert a string representation of a number of bytes into
    an integer, and also to print out human-readable strings representing a number of bytes.

    In conformance with [IEC 80000-13 Standard](https://en.wikipedia.org/wiki/ISO/IEC_80000) we interpret `'1KB'` to mean 1000 bytes,
    and `'1KiB'` to mean 1024 bytes. In general, including a middle `'i'` will cause the unit to be interpreted as a power of 2,
    rather than a power of 10 (so, for example, `'1 MB'` is treated as `1_000_000` bytes, whereas `'1 MiB'` is treated as `1_048_576` bytes).

    !!! info
        Note that `1b` will be parsed as "1 byte" and not "1 bit".

    ```py
    from pydantic import BaseModel, ByteSize

    class MyModel(BaseModel):
        size: ByteSize

    print(MyModel(size=52000).size)
    #> 52000
    print(MyModel(size='3000 KiB').size)
    #> 3072000

    m = MyModel(size='50 PB')
    print(m.size.human_readable())
    #> 44.4PiB
    print(m.size.human_readable(decimal=True))
    #> 50.0PB

    print(m.size.to('TiB'))
    #> 45474.73508864641
    ```
    """
    byte_sizes = {'b': 1, 'kb': 10 ** 3, 'mb': 10 ** 6, 'gb': 10 ** 9, 'tb': 10 ** 12, 'pb': 10 ** 15, 'eb': 10 ** 18, 'kib': 2 ** 10, 'mib': 2 ** 20, 'gib': 2 ** 30, 'tib': 2 ** 40, 'pib': 2 ** 50, 'eib': 2 ** 60, 'bit': 1 / 8, 'kbit': 10 ** 3 / 8, 'mbit': 10 ** 6 / 8, 'gbit': 10 ** 9 / 8, 'tbit': 10 ** 12 / 8, 'pbit': 10 ** 15 / 8, 'ebit': 10 ** 18 / 8, 'kibit': 2 ** 10 / 8, 'mibit': 2 ** 20 / 8, 'gibit': 2 ** 30 / 8, 'tibit': 2 ** 40 / 8, 'pibit': 2 ** 50 / 8, 'eibit': 2 ** 60 / 8}
    byte_sizes.update({k.lower()[0]: v for k, v in byte_sizes.items() if 'i' not in k})
    byte_string_pattern = '^\\s*(\\d*\\.?\\d+)\\s*(\\w+)?'
    byte_string_re = re.compile(byte_string_pattern, re.IGNORECASE)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(function=cls._validate, schema=core_schema.union_schema([core_schema.str_schema(pattern=cls.byte_string_pattern), core_schema.int_schema(ge=0)], custom_error_type='byte_size', custom_error_message='could not parse value and unit from byte string'), serialization=core_schema.plain_serializer_function_ser_schema(int, return_schema=core_schema.int_schema(ge=0)))

    @classmethod
    def _validate(cls, __input_value: Any, _: core_schema.ValidationInfo) -> ByteSize:
        try:
            return cls(int(__input_value))
        except ValueError:
            pass
        str_match = cls.byte_string_re.match(str(__input_value))
        if str_match is None:
            raise PydanticCustomError('byte_size', 'could not parse value and unit from byte string')
        scalar, unit = str_match.groups()
        if unit is None:
            unit = 'b'
        try:
            unit_mult = cls.byte_sizes[unit.lower()]
        except KeyError:
            raise PydanticCustomError('byte_size_unit', 'could not interpret byte unit: {unit}', {'unit': unit})
        return cls(int(float(scalar) * unit_mult))

    def human_readable(self, decimal: bool=False) -> str:
        """Converts a byte size to a human readable string.

        Args:
            decimal: If True, use decimal units (e.g. 1000 bytes per KB). If False, use binary units
                (e.g. 1024 bytes per KiB).

        Returns:
            A human readable string representation of the byte size.
        """
        if decimal:
            divisor = 1000
            units = ('B', 'KB', 'MB', 'GB', 'TB', 'PB')
            final_unit = 'EB'
        else:
            divisor = 1024
            units = ('B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB')
            final_unit = 'EiB'
        num = float(self)
        for unit in units:
            if abs(num) < divisor:
                if unit == 'B':
                    return f'{num:0.0f}{unit}'
                else:
                    return f'{num:0.1f}{unit}'
            num /= divisor
        return f'{num:0.1f}{final_unit}'

    def to(self, unit: str) -> float:
        """Converts a byte size to another unit, including both byte and bit units.

        Args:
            unit: The unit to convert to. Must be one of the following: B, KB, MB, GB, TB, PB, EB,
                KiB, MiB, GiB, TiB, PiB, EiB (byte units) and
                bit, kbit, mbit, gbit, tbit, pbit, ebit,
                kibit, mibit, gibit, tibit, pibit, eibit (bit units).

        Returns:
            The byte size in the new unit.
        """
        try:
            unit_div = self.byte_sizes[unit.lower()]
        except KeyError:
            raise PydanticCustomError('byte_size_unit', 'Could not interpret byte unit: {unit}', {'unit': unit})
        return self / unit_div