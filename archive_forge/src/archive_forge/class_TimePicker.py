from __future__ import annotations
import logging # isort:skip
from ...core.enums import CalendarPosition
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from .inputs import InputWidget
class TimePicker(PickerBase, TimeCommon):
    """ Widget for picking time. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    value = Nullable(Time, default=None, help='\n    The initial or picked time.\n    ')
    time_format = String(default='H:i', help='\n    Formatting specification for the display of the picked date.\n\n    +---+------------------------------------+------------+\n    | H | Hours (24 hours)                   | 00 to 23   |\n    | h | Hours                              | 1 to 12    |\n    | G | Hours, 2 digits with leading zeros | 1 to 12    |\n    | i | Minutes                            | 00 to 59   |\n    | S | Seconds, 2 digits                  | 00 to 59   |\n    | s | Seconds                            | 0, 1 to 59 |\n    | K | AM/PM                              | AM or PM   |\n    +---+------------------------------------+------------+\n\n    See also https://flatpickr.js.org/formatting/#date-formatting-tokens.\n    ')
    min_time = Nullable(Time)(default=None, help='\n    Optional earliest allowable time.\n    ')
    max_time = Nullable(Time)(default=None, help='\n    Optional latest allowable time.\n    ')