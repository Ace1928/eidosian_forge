from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
def _session_callback_removed(self, event: SessionCallbackRemoved) -> None:
    ...