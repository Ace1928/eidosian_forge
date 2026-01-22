from __future__ import annotations
import logging # isort:skip
from ...core.enums import CoordinateUnits
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property_mixins import FillProps, LineProps
from ..graphics import Marking
from .annotation import DataAnnotation
@abstract
class ArrowHead(Marking):
    """ Base class for arrow heads.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    size = NumberSpec(default=25, help='\n    The size, in pixels, of the arrow head.\n    ')