from __future__ import annotations
import dataclasses
import enum
import typing
class VAlign(str, enum.Enum):
    """Filler alignment"""
    TOP = 'top'
    MIDDLE = 'middle'
    BOTTOM = 'bottom'