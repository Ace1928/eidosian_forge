from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class ModelSchema(TypedDict, total=False):
    type: Required[Literal['model']]
    cls: Required[Type[Any]]
    schema: Required[CoreSchema]
    custom_init: bool
    root_model: bool
    post_init: str
    revalidate_instances: Literal['always', 'never', 'subclass-instances']
    strict: bool
    frozen: bool
    extra_behavior: ExtraBehavior
    config: CoreConfig
    ref: str
    metadata: Any
    serialization: SerSchema