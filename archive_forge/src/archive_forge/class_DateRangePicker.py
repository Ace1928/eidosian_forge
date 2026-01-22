from __future__ import annotations
import logging # isort:skip
from ...core.enums import CalendarPosition
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from .inputs import InputWidget
class DateRangePicker(BaseDatePicker):
    """ Calendar-based picker of date ranges. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    value = Nullable(Tuple(Date, Date), default=None, help='\n    The initial or picked date range.\n    ')