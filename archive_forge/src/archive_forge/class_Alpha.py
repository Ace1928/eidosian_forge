from __future__ import annotations
import logging # isort:skip
import re
from typing import Any
from ... import colors
from .. import enums
from .bases import Init, Property
from .container import Tuple
from .either import Either
from .enum import Enum
from .numeric import Byte, Percent
from .singletons import Undefined
from .string import Regex
class Alpha(Percent):

    def __init__(self, default: Init[float]=1.0, *, help: str | None=None) -> None:
        help = f'{help or ''}\n{ALPHA_DEFAULT_HELP}'
        super().__init__(default=default, help=help)