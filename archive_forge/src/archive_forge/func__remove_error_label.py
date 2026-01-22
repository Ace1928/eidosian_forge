from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
def _remove_error_label(self, label: str) -> None:
    """Remove the given label from this error."""
    self._error_labels.discard(label)