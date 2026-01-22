from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
class ChildRecord(NamedTuple):
    """Child record as a NamedTuple."""
    type: ChildType
    kwargs: Dict[str, Any]
    dg: DeltaGenerator