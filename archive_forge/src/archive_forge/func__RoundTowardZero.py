import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def _RoundTowardZero(value, divider):
    """Truncates the remainder part after division."""
    result = value // divider
    remainder = value % divider
    if result < 0 and remainder > 0:
        return result + 1
    else:
        return result