from __future__ import annotations
import logging # isort:skip
from typing import (
class RootAdded(TypedDict):
    kind: Literal['RootAdded']
    model: Ref