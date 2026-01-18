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
def tobitmap(self, name='image'):
    """
        Returns the image converted to an X11 bitmap.

        .. note:: This method only works for mode "1" images.

        :param name: The name prefix to use for the bitmap variables.
        :returns: A string containing an X11 bitmap.
        :raises ValueError: If the mode is not "1"
        """
    self.load()
    if self.mode != '1':
        msg = 'not a bitmap'
        raise ValueError(msg)
    data = self.tobytes('xbm')
    return b''.join([f'#define {name}_width {self.size[0]}\n'.encode('ascii'), f'#define {name}_height {self.size[1]}\n'.encode('ascii'), f'static char {name}_bits[] = {{\n'.encode('ascii'), data, b'};'])