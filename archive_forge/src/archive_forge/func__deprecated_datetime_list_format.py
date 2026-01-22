from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from ..util.deprecation import deprecated
from ..util.strings import format_docstring
from ..util.warnings import warn
from .tickers import Ticker
def _deprecated_datetime_list_format(fmt: list[str]) -> str:
    deprecated('Passing lists of formats for DatetimeTickFormatter scales was deprecated in Bokeh 3.0. Configure a single string format for each scale')
    if len(fmt) == 0:
        raise ValueError('Datetime format list must contain one element')
    if len(fmt) > 1:
        warn(f'DatetimeFormatter scales now only accept a single format. Using the first provided: {fmt[0]!r}')
    return fmt[0]