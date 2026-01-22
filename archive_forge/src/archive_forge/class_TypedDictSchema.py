from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class TypedDictSchema(TypedDict, total=False):
    type: Required[Literal['typed-dict']]
    fields: Required[Dict[str, TypedDictField]]
    computed_fields: List[ComputedField]
    strict: bool
    extras_schema: CoreSchema
    extra_behavior: ExtraBehavior
    total: bool
    populate_by_name: bool
    ref: str
    metadata: Any
    serialization: SerSchema
    config: CoreConfig