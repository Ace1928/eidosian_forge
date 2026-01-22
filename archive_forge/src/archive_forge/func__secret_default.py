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
@default('secret')
def _secret_default(self):
    if Path(self.secret_file).exists():
        with Path(self.secret_file).open('rb') as f:
            return f.read()
    else:
        secret = encodebytes(os.urandom(1024))
        self._write_secret_file(secret)
        return secret