from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class Pinch(PointEvent):
    """ Announce a pinch event on a Bokeh plot.

    Attributes:
        scale (float) : the (signed) amount of scaling
        sx (float) : x-coordinate of the event in *screen* space
        sy (float) : y-coordinate of the event in *screen* space
        x (float) : x-coordinate of the event in *data* space
        y (float) : y-coordinate of the event in *data* space

    .. note::
        This event is only applicable for touch-enabled devices.

    """
    event_name = 'pinch'

    def __init__(self, model: Plot | None, *, scale: float | None=None, sx: float | None=None, sy: float | None=None, x: float | None=None, y: float | None=None, modifiers: KeyModifiers | None=None):
        self.scale = scale
        super().__init__(model, sx=sx, sy=sy, x=x, y=y, modifiers=modifiers)