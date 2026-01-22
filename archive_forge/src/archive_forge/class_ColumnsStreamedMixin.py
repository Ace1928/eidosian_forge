from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class ColumnsStreamedMixin:

    def _columns_streamed(self, event: ColumnsStreamedEvent) -> None:
        ...