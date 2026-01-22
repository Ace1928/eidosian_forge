from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class EnumSchema(TypedDict, total=False):
    type: Required[Literal['enum']]
    cls: Required[Any]
    members: Required[List[Any]]
    sub_type: Literal['str', 'int', 'float']
    missing: Callable[[Any], Any]
    strict: bool
    ref: str
    metadata: Any
    serialization: SerSchema