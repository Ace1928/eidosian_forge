from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class EventRep(TypedDict):
    type: Literal['event']
    name: str
    values: dict[str, Any]