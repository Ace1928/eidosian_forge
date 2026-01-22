from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .expressions import CoordinateTransform
class NodeCoordinates(GraphCoordinates):
    """
    Node coordinate expression obtained from ``LayoutProvider``

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)