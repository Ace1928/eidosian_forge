from __future__ import annotations
import functools
import time
from collections import deque
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import Any, Callable, Deque, MutableMapping, Optional, TypeVar, cast
from pymongo.write_concern import WriteConcern
def apply_write_concern(cmd: MutableMapping[str, Any], write_concern: Optional[WriteConcern]) -> None:
    """Apply the given write concern to a command."""
    if not write_concern or write_concern.is_server_default:
        return
    wc = write_concern.document
    if get_timeout() is not None:
        wc.pop('wtimeout', None)
    if wc:
        cmd['writeConcern'] = wc