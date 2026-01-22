from __future__ import annotations
import enum
import sys
import types
import typing
import warnings
def _extract_buffer_length(obj: typing.Any) -> typing.Tuple[typing.Any, int]:
    from cryptography.hazmat.bindings._rust import _openssl
    buf = _openssl.ffi.from_buffer(obj)
    return (buf, int(_openssl.ffi.cast('uintptr_t', buf)))