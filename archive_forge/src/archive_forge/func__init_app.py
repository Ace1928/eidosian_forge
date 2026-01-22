from __future__ import annotations
import sys
from typing import TYPE_CHECKING, ClassVar
import param
from ..io.state import state
from ..viewable import Viewable
from ..widgets import Terminal
from .base import PaneBase
def _init_app(self, comm):
    if self.object is None or self.object.is_running:
        return
    state.execute(self._run_app)