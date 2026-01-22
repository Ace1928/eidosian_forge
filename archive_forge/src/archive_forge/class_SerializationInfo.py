from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class SerializationInfo(Protocol):

    @property
    def include(self) -> IncExCall:
        ...

    @property
    def exclude(self) -> IncExCall:
        ...

    @property
    def mode(self) -> str:
        ...

    @property
    def by_alias(self) -> bool:
        ...

    @property
    def exclude_unset(self) -> bool:
        ...

    @property
    def exclude_defaults(self) -> bool:
        ...

    @property
    def exclude_none(self) -> bool:
        ...

    @property
    def round_trip(self) -> bool:
        ...

    def mode_is_json(self) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...