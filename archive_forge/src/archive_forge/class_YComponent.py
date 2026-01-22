from __future__ import annotations
import logging # isort:skip
from math import inf
from ..core.enums import Direction
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class YComponent(XYComponent):
    """ Y-component of a coordinate system transform to cartesian coordinates. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)