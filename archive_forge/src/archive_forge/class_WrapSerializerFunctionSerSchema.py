from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class WrapSerializerFunctionSerSchema(TypedDict, total=False):
    type: Required[Literal['function-wrap']]
    function: Required[WrapSerializerFunction]
    is_field_serializer: bool
    info_arg: bool
    schema: CoreSchema
    return_schema: CoreSchema
    when_used: WhenUsed