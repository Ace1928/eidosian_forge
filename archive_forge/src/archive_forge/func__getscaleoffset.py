from __future__ import annotations
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
from . import (
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
def _getscaleoffset(expr):
    a = expr(_E(1, 0))
    return (a.scale, a.offset) if isinstance(a, _E) else (0, a)