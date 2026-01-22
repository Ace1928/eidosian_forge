from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class Pan(PointEvent):
    """ Announce a pan event on a Bokeh plot.

    Attributes:
        delta_x (float) : the amount of scroll in the x direction
        delta_y (float) : the amount of scroll in the y direction
        direction (float) : the direction of scroll (1 or -1)
        sx (float) : x-coordinate of the event in *screen* space
        sy (float) : y-coordinate of the event in *screen* space
        x (float) : x-coordinate of the event in *data* space
        y (float) : y-coordinate of the event in *data* space

    """
    event_name = 'pan'

    def __init__(self, model: Plot | None, *, delta_x: float | None=None, delta_y: float | None=None, direction: Literal[-1, -1] | None=None, sx: float | None=None, sy: float | None=None, x: float | None=None, y: float | None=None, modifiers: KeyModifiers | None=None):
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.direction = direction
        super().__init__(model, sx=sx, sy=sy, x=x, y=y, modifiers=modifiers)