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
def writeCountReference(self, table, name, size=2, value=None):
    ref = CountReference(table, name, size=size, value=value)
    self.items.append(ref)
    return ref