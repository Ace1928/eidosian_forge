from __future__ import annotations
import logging # isort:skip
from types import SimpleNamespace
from typing import Any, Generic, TypeVar
from .bases import ParameterizedProperty, Property
@property
def type_params(self):
    return list(self._fields.values())