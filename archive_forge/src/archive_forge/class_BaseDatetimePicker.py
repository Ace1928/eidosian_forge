from __future__ import annotations
import logging # isort:skip
from ...core.enums import CalendarPosition
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from .inputs import InputWidget
@abstract
class BaseDatetimePicker(PickerBase, DateCommon, TimeCommon):
    """ Bases for various calendar-based datetime picker widgets.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    min_date = Nullable(Either(Datetime, Date), default=None, help='\n    Optional earliest allowable date and time.\n    ')
    max_date = Nullable(Either(Datetime, Date), default=None, help='\n    Optional latest allowable date and time.\n    ')
    date_format = Override(default='Y-m-d H:i')