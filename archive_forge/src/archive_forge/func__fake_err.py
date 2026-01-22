from __future__ import annotations
import contextlib
import errno
import ipaddress
import os
import socket
import sys
from typing import (
import attrs
import trio
from trio._util import NoPublicConstructor, final
def _fake_err(code: int) -> NoReturn:
    raise OSError(code, os.strerror(code))