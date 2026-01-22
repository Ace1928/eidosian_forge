from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class RangesUpdate(PlotEvent):
    """ Announce combined range updates in a single event.

    Attributes:
        x0 (float) : start x-coordinate for the default x-range
        x1 (float) : end x-coordinate for the default x-range
        y0 (float) : start x-coordinate for the default y-range
        y1 (float) : end y-coordinate for the default x-range

    Callbacks may be added to range ``start`` and ``end`` properties to respond
    to range changes, but this can result in multiple callbacks being invoked
    for a single logical operation (e.g. a pan or zoom). This event is emitted
    by supported tools when the entire range update is complete, in order to
    afford a *single* event that can be responded to.

    """
    event_name = 'rangesupdate'

    def __init__(self, model: Plot | None, *, x0: float | None=None, x1: float | None=None, y0: float | None=None, y1: float | None=None):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        super().__init__(model=model)