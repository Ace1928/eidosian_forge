from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class NoOverlap(LabelingPolicy):
    """ Basic labeling policy avoiding label overlap. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    min_distance = Int(default=5, help='\n    Minimum distance between labels in pixels.\n    ')