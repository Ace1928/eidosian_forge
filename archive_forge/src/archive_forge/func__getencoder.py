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
def _getencoder(mode, encoder_name, args, extra=()):
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)
    try:
        encoder = ENCODERS[encoder_name]
    except KeyError:
        pass
    else:
        return encoder(mode, *args + extra)
    try:
        encoder = getattr(core, encoder_name + '_encoder')
    except AttributeError as e:
        msg = f'encoder {encoder_name} not available'
        raise OSError(msg) from e
    return encoder(mode, *args + extra)