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
def _DATETIME_TICK_FORMATTER_HELP(field: str) -> str:
    return f'\n    Formats for displaying datetime values in the {field} range.\n\n    See the :class:`~bokeh.models.formatters.DatetimeTickFormatter` help for a list of all supported formats.\n    '