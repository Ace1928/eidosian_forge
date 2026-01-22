from __future__ import annotations
import logging # isort:skip
from ..core.enums import JitterRandomDistribution, StepMode
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .sources import ColumnarDataSource
class LinearInterpolator(Interpolator):
    """ Compute a linear interpolation between the control points provided through
    the ``x``, ``y``, and ``data`` parameters.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)