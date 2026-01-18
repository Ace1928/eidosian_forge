import datetime
from typing import Any, Optional, cast
from dateutil.rrule import WEEKLY, rrule
from arrow.constants import (
def validate_ordinal(value: Any) -> None:
    """Raise an exception if value is an invalid Gregorian ordinal.

    :param value: the input to be checked

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'Ordinal must be an integer (got type {type(value)}).')
    if not MIN_ORDINAL <= value <= MAX_ORDINAL:
        raise ValueError(f'Ordinal {value} is out of range.')