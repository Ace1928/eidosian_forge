from __future__ import annotations
import logging # isort:skip
from typing import (
class RootRemoved(TypedDict):
    kind: Literal['RootRemoved']
    model: Ref