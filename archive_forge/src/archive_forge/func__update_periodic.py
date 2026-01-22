from __future__ import annotations
import os
from fnmatch import fnmatch
from typing import (
import param
from ..io import PeriodicCallback
from ..layout import (
from ..util import fullpath
from ..viewable import Layoutable
from .base import CompositeWidget
from .button import Button
from .input import TextInput
from .select import CrossSelector
def _update_periodic(self, event: param.parameterized.Event):
    if event.new:
        self._periodic.period = event.new
        if not self._periodic.running:
            self._periodic.start()
    elif self._periodic.running:
        self._periodic.stop()