from __future__ import annotations
import logging # isort:skip
from math import inf
from typing import Any as any
from ...core.has_props import abstract
from ...core.properties import (
from ...util.deprecation import deprecated
from ..dom import HTML
from ..formatters import TickFormatter
from ..ui import Tooltip
from .widget import Widget
def ColorMap(*args: any, **kwargs: any) -> PaletteSelect:
    """ Color palette select widget.

    .. deprecated:: 3.4.0
        Use ``PaletteSelect`` widget instead.
    """
    deprecated((3, 4, 0), 'ColorMap widget', 'PaletteSelect widget')
    return PaletteSelect(*args, **kwargs)