from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class TypedDictField(TypedDict, total=False):
    type: Required[Literal['typed-dict-field']]
    schema: Required[CoreSchema]
    required: bool
    validation_alias: Union[str, List[Union[str, int]], List[List[Union[str, int]]]]
    serialization_alias: str
    serialization_exclude: bool
    metadata: Any