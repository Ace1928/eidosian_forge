from __future__ import annotations
import sys
from typing import TYPE_CHECKING, ClassVar
import param
from ..io.state import state
from ..viewable import Viewable
from ..widgets import Terminal
from .base import PaneBase

    The `Textual` pane provides a wrapper around a Textual App component,
    rendering it inside a Terminal and running it on the existing Panel
    event loop, i.e. either on the server or the notebook asyncio.EventLoop.

    Reference: https://panel.holoviz.org/reference/panes/Textual.html

    :Example:

    >>> Textual(app)
    