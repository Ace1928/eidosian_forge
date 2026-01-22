from __future__ import annotations
import logging # isort:skip
import numbers
from datetime import date, datetime, timezone
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.descriptors import UnsetValueError
from ...core.property.singletons import Undefined
from ...core.validation import error
from ...core.validation.errors import EQUAL_SLIDER_START_END
from ..formatters import TickFormatter
from .widget import Widget
class RangeSlider(NumericalSlider):
    """ Range-slider based number range selection widget. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    value = Required(Tuple(Float, Float), help='\n    Initial or selected range.\n    ')
    value_throttled = Readonly(Required(Tuple(Float, Float)), help='\n    Initial or selected value, throttled according to report only on mouseup.\n    ')
    start = Required(Float, help='\n    The minimum allowable value.\n    ')
    end = Required(Float, help='\n    The maximum allowable value.\n    ')
    step = Float(default=1, help='\n    The step between consecutive values.\n    ')
    format = Override(default='0[.]00')