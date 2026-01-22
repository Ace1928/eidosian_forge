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
class BasicTickFormatter(TickFormatter):
    """ Display tick values from continuous ranges as "basic numbers",
    using scientific notation when appropriate by default.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    precision = Either(Auto, Int, help='\n    How many digits of precision to display in tick labels.\n    ')
    use_scientific = Bool(True, help='\n    Whether to ever display scientific notation. If ``True``, then\n    when to use scientific notation is controlled by ``power_limit_low``\n    and ``power_limit_high``.\n    ')
    power_limit_high = Int(5, help='\n    Limit the use of scientific notation to when::\n\n        log(x) >= power_limit_high\n\n    ')
    power_limit_low = Int(-3, help='\n    Limit the use of scientific notation to when::\n\n        log(x) <= power_limit_low\n\n    ')