from __future__ import annotations
import io
from functools import partial
from typing import (
import trio
from ._util import async_wraps
from .abc import AsyncResource
class _HasMode(Protocol):

    @property
    def mode(self) -> str:
        ...