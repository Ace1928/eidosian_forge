from __future__ import annotations
import logging
import re
import ssl
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, Tuple, TypeVar
from .. import (
from .._core._typedattr import TypedAttributeSet, typed_attribute
from ..abc import AnyByteStream, ByteStream, Listener, TaskGroup
class TLSAttribute(TypedAttributeSet):
    """Contains Transport Layer Security related attributes."""
    alpn_protocol: str | None = typed_attribute()
    channel_binding_tls_unique: bytes = typed_attribute()
    cipher: tuple[str, str, int] = typed_attribute()
    peer_certificate: None | dict[str, str | _PCTRTTT | _PCTRTT] = typed_attribute()
    peer_certificate_binary: bytes | None = typed_attribute()
    server_side: bool = typed_attribute()
    shared_ciphers: list[tuple[str, str, int]] | None = typed_attribute()
    ssl_object: ssl.SSLObject = typed_attribute()
    standard_compatible: bool = typed_attribute()
    tls_version: str = typed_attribute()