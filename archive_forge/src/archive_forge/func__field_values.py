from __future__ import annotations
from abc import ABC
from collections.abc import Sequence
from dataclasses import fields
from typing import Any, Dict
from numpy import ndarray
@property
def _field_values(self) -> tuple:
    """Tuple of field values in any inheriting result dataclass."""
    return tuple((getattr(self, name) for name in self._field_names))