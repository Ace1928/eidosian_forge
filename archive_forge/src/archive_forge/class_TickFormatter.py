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
@abstract
class TickFormatter(Model):
    """ A base class for all tick formatter types.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)