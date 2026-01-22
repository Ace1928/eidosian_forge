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
class OTLOffsetOverflowError(Exception):

    def __init__(self, overflowErrorRecord):
        self.value = overflowErrorRecord

    def __str__(self):
        return repr(self.value)