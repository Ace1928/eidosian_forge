from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class DocumentPatchedMixin:

    def _document_patched(self, event: DocumentPatchedEvent) -> None:
        ...