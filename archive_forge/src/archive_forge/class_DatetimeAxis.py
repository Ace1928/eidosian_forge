from __future__ import annotations
import logging # isort:skip
from ..core.enums import Align, LabelOrientation
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_mixins import ScalarFillProps, ScalarLineProps, ScalarTextProps
from .formatters import (
from .labeling import AllLabels, LabelingPolicy
from .renderers import GuideRenderer
from .tickers import (
class DatetimeAxis(LinearAxis):
    """ A ``LinearAxis`` that picks nice numbers for tick locations on
    a datetime scale. Configured with a ``DatetimeTickFormatter`` by
    default.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    ticker = Override(default=InstanceDefault(DatetimeTicker))
    formatter = Override(default=InstanceDefault(DatetimeTickFormatter))