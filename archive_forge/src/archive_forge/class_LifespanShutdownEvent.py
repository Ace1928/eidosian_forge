from __future__ import annotations
import sys
import types
from typing import (
class LifespanShutdownEvent(TypedDict):
    type: Literal['lifespan.shutdown']