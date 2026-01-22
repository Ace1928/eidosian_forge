from __future__ import annotations
import logging # isort:skip
from typing import (
class ColumnDataChanged(TypedDict):
    kind: Literal['ColumnDataChanged']
    model: Ref
    attr: str
    data: DataDict
    cols: list[str] | None