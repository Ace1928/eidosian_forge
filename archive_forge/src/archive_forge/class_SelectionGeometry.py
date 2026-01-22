from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class SelectionGeometry(PlotEvent):
    """ Announce the coordinates of a selection event on a plot.

    Attributes:
        geometry (dict) : a dictionary containing the coordinates of the
            selection event.
        final (bool) : whether the selection event is the last selection event
            in the case of selections on every mousemove.

    """
    event_name = 'selectiongeometry'

    def __init__(self, model: Plot | None, geometry: GeometryData | None=None, final: bool=True) -> None:
        self.geometry = geometry
        self.final = final
        super().__init__(model=model)