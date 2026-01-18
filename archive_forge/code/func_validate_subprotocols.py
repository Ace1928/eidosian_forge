from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def validate_subprotocols(subprotocols: Sequence[Subprotocol]) -> None:
    """
    Validate that ``subprotocols`` is suitable for :func:`build_subprotocol`.

    """
    if not isinstance(subprotocols, Sequence):
        raise TypeError('subprotocols must be a list')
    if isinstance(subprotocols, str):
        raise TypeError('subprotocols must be a list, not a str')
    for subprotocol in subprotocols:
        if not _token_re.fullmatch(subprotocol):
            raise ValueError(f'invalid subprotocol: {subprotocol}')