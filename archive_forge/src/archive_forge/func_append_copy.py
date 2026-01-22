from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
def append_copy(self, other: MutableExpander) -> None:
    """Append a copy of another MutableExpander's children to this
        MutableExpander.
        """
    other_records = other._child_records.copy()
    for record in other_records:
        self._create_child(record.type, record.kwargs)