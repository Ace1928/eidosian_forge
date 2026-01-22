from __future__ import annotations
import enum
import typing
from urwid.canvas import CompositeCanvas, SolidCanvas
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget import Widget

        Render the divider as a canvas and return it.

        >>> Divider().render((10,)).text # ... = b in Python 3
        [...'          ']
        >>> Divider(u'-', top=1).render((10,)).text
        [...'          ', ...'----------']
        >>> Divider(u'x', bottom=2).render((5,)).text
        [...'xxxxx', ...'     ', ...'     ']
        