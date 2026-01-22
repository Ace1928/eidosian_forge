from __future__ import annotations
import logging # isort:skip
from ...core.enums import CalendarPosition
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from .inputs import InputWidget
@abstract
class PickerBase(InputWidget):
    """ Base class for various kinds of picker widgets. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    position = Enum(CalendarPosition, default='auto', help='\n    Where the calendar is rendered relative to the input when ``inline`` is False.\n    ')
    inline = Bool(default=False, help='\n    Whether the calendar sholud be displayed inline.\n    ')