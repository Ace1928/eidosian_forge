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
class PrintfTickFormatter(TickFormatter):
    """ Tick formatter based on a printf-style format string. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    format = String('%s', help="\n    The number format, as defined as follows: the placeholder in the format\n    string is marked by % and is followed by one or more of these elements,\n    in this order:\n\n    * An optional ``+`` sign\n        Causes the result to be preceded with a plus or minus sign on numeric\n        values. By default, only the ``-`` sign is used on negative numbers.\n\n    * An optional padding specifier\n        Specifies what (if any) character to use for padding. Possible values\n        are 0 or any other character preceded by a ``'`` (single quote). The\n        default is to pad with spaces.\n\n    * An optional ``-`` sign\n        Causes sprintf to left-align the result of this placeholder. The default\n        is to right-align the result.\n\n    * An optional number\n        Specifies how many characters the result should have. If the value to be\n        returned is shorter than this number, the result will be padded.\n\n    * An optional precision modifier\n        Consists of a ``.`` (dot) followed by a number, specifies how many digits\n        should be displayed for floating point numbers. When used on a string, it\n        causes the result to be truncated.\n\n    * A type specifier\n        Can be any of:\n\n        - ``%`` --- yields a literal ``%`` character\n        - ``b`` --- yields an integer as a binary number\n        - ``c`` --- yields an integer as the character with that ASCII value\n        - ``d`` or ``i`` --- yields an integer as a signed decimal number\n        - ``e`` --- yields a float using scientific notation\n        - ``u`` --- yields an integer as an unsigned decimal number\n        - ``f`` --- yields a float as is\n        - ``o`` --- yields an integer as an octal number\n        - ``s`` --- yields a string as is\n        - ``x`` --- yields an integer as a hexadecimal number (lower-case)\n        - ``X`` --- yields an integer as a hexadecimal number (upper-case)\n\n    ")