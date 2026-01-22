from __future__ import annotations
from enum import Enum
from .data_structures import Point
class MouseModifier(Enum):
    SHIFT = 'SHIFT'
    ALT = 'ALT'
    CONTROL = 'CONTROL'