from __future__ import annotations
import sys
import types
from typing import (
class LifespanShutdownFailedEvent(TypedDict):
    type: Literal['lifespan.shutdown.failed']
    message: str