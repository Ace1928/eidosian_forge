from __future__ import annotations
import hashlib
import os
import sys
import typing as t
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from hmac import HMAC
from pathlib import Path
from base64 import encodebytes
from jupyter_core.application import JupyterApp, base_flags
from traitlets import Any, Bool, Bytes, Callable, Enum, Instance, Integer, Unicode, default, observe
from traitlets.config import LoggingConfigurable, MultipleInstanceError
from . import NO_CONVERT, __version__, read, reads
def yield_everything(obj):
    """Yield every item in a container as bytes

    Allows any JSONable object to be passed to an HMAC digester
    without having to serialize the whole thing.
    """
    if isinstance(obj, dict):
        for key in sorted(obj):
            value = obj[key]
            assert isinstance(key, str)
            yield key.encode()
            yield from yield_everything(value)
    elif isinstance(obj, (list, tuple)):
        for element in obj:
            yield from yield_everything(element)
    elif isinstance(obj, str):
        yield obj.encode('utf8')
    else:
        yield str(obj).encode('utf8')