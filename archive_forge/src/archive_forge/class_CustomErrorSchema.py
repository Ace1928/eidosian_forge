from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class CustomErrorSchema(TypedDict, total=False):
    type: Required[Literal['custom-error']]
    schema: Required[CoreSchema]
    custom_error_type: Required[str]
    custom_error_message: str
    custom_error_context: Dict[str, Union[str, int, float]]
    ref: str
    metadata: Any
    serialization: SerSchema