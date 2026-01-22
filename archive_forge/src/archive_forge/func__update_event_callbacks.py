from __future__ import annotations
import logging # isort:skip
from collections import defaultdict
from inspect import signature
from typing import (
from ..events import Event, ModelEvent
from ..util.functions import get_param_info
def _update_event_callbacks(self) -> None:
    if self.document is None:
        return
    for key in self._event_callbacks:
        from ..model import Model
        self.document.callbacks.subscribe(key, cast(Model, self))