from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class DifferenceFilter(CompositeFilter):
    """ Computes difference of indices resulting from other filters. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)