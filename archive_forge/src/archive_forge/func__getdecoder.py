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
def _getdecoder(mode, decoder_name, args, extra=()):
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)
    try:
        decoder = DECODERS[decoder_name]
    except KeyError:
        pass
    else:
        return decoder(mode, *args + extra)
    try:
        decoder = getattr(core, decoder_name + '_decoder')
    except AttributeError as e:
        msg = f'decoder {decoder_name} not available'
        raise OSError(msg) from e
    return decoder(mode, *args + extra)