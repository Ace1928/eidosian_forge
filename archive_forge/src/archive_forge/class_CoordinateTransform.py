from __future__ import annotations
import logging # isort:skip
from math import inf
from ..core.enums import Direction
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
@abstract
class CoordinateTransform(Expression):
    """ Base class for coordinate transforms. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def x(self):
        return XComponent(transform=self)

    @property
    def y(self):
        return YComponent(transform=self)