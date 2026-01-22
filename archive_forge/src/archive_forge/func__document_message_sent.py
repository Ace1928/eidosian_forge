from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
def _document_message_sent(self, event: MessageSentEvent) -> None:
    ...