from fontTools.config import OPTIONS
from fontTools.misc.textTools import Tag, bytesjoin
from .DefaultTable import DefaultTable
from enum import IntEnum
import sys
import array
import struct
import logging
from functools import lru_cache
from typing import Iterator, NamedTuple, Optional, Tuple
def writeUInt24(self, value):
    assert 0 <= value < 16777216, value
    b = struct.pack('>L', value)
    self.items.append(b[1:])