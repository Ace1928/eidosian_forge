import datetime
from typing import Any, Optional, cast
from dateutil.rrule import WEEKLY, rrule
from arrow.constants import (
def validate_bounds(bounds: str) -> None:
    if bounds != '()' and bounds != '(]' and (bounds != '[)') and (bounds != '[]'):
        raise ValueError("Invalid bounds. Please select between '()', '(]', '[)', or '[]'.")