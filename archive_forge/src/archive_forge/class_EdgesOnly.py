from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .expressions import CoordinateTransform
class EdgesOnly(GraphHitTestPolicy):
    """
    With the ``EdgesOnly`` policy, only graph edges are able to be selected and
    inspected. There is no selection or inspection of graph nodes.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)