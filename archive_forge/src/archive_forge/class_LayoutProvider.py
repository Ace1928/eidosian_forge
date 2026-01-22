from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .expressions import CoordinateTransform
@abstract
class LayoutProvider(Model):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def node_coordinates(self) -> NodeCoordinates:
        return NodeCoordinates(layout=self)

    @property
    def edge_coordinates(self) -> EdgeCoordinates:
        return EdgeCoordinates(layout=self)