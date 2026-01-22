from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, List, Optional
def _convert_row_as_tuple(self, row: Row) -> tuple:
    return tuple(map(str, row.asDict().values()))