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
class OffsetToWriter(object):

    def __init__(self, subWriter, offsetSize):
        self.subWriter = subWriter
        self.offsetSize = offsetSize

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.subWriter == other.subWriter and self.offsetSize == other.offsetSize

    def __hash__(self):
        return hash((self.subWriter, self.offsetSize))