from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class PointEvent(PlotEvent):
    """ Base class for UI events associated with a specific (x,y) point.

    Attributes:
        sx (float) : x-coordinate of the event in *screen* space
        sy (float) : y-coordinate of the event in *screen* space
        x (float) : x-coordinate of the event in *data* space
        y (float) : y-coordinate of the event in *data* space

    Note that data space coordinates are relative to the default range, not
    any extra ranges, and the the screen space origin is at the top left of
    the HTML canvas.

    """

    def __init__(self, model: Plot | None, sx: float | None=None, sy: float | None=None, x: float | None=None, y: float | None=None, modifiers: KeyModifiers | None=None):
        self.sx = sx
        self.sy = sy
        self.x = x
        self.y = y
        self.modifiers = modifiers
        super().__init__(model=model)