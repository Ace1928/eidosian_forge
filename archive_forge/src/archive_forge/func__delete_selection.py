from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, Optional, Pattern
import re
import time
from pyglet import clock
from pyglet import event
from pyglet.window import key
def _delete_selection(self) -> None:
    start = min(self._mark, self._position)
    end = max(self._mark, self._position)
    self._position = start
    self._mark = None
    self._layout.document.delete_text(start, end)
    self._layout.set_selection(0, 0)