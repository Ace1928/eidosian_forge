from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any
from ..core.types import ID
class ErrorReason(Enum):
    NO_ERROR = auto()
    HTTP_ERROR = auto()
    NETWORK_ERROR = auto()