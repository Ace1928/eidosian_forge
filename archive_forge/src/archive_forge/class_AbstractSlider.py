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
@abstract
class AbstractSlider(Widget):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            self.lookup('value_throttled')._set(self, Undefined, self.value)
        except UnsetValueError:
            pass
        except AttributeError:
            pass
    orientation = Enum('horizontal', 'vertical', help='\n    Orient the slider either horizontally (default) or vertically.\n    ')
    title = Nullable(String, default='', help="\n    The slider's label (supports :ref:`math text <ug_styling_mathtext>`).\n    ")
    show_value = Bool(default=True, help="\n    Whether or not show slider's value.\n    ")
    direction = Enum('ltr', 'rtl', help='\n    ')
    tooltips = Bool(default=True, help="\n    Display the slider's current value in a tooltip.\n    ")
    bar_color = Color(default='#e6e6e6', help='\n    ')
    width = Override(default=300)

    @error(EQUAL_SLIDER_START_END)
    def _check_missing_dimension(self):
        if hasattr(self, 'start') and hasattr(self, 'end'):
            if self.start == self.end:
                return f'{self!s} with title {self.title!s}'