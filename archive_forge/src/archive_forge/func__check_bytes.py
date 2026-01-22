from __future__ import annotations
import enum
import sys
import types
import typing
import warnings
def _check_bytes(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f'{name} must be bytes')