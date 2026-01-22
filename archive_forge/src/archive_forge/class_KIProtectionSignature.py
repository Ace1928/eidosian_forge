from __future__ import annotations
import inspect
import signal
import sys
from functools import wraps
from typing import TYPE_CHECKING, Final, Protocol, TypeVar
import attrs
from .._util import is_main_thread
class KIProtectionSignature(Protocol):
    __name__: str

    def __call__(self, f: CallableT, /) -> CallableT:
        pass