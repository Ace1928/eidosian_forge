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
def writeValueRecord(self, writer, font, valueRecord):
    for name, isDevice, signed in self.format:
        value = getattr(valueRecord, name, 0)
        if isDevice:
            if value:
                subWriter = writer.getSubWriter()
                writer.writeSubTable(subWriter, offsetSize=2)
                value.compile(subWriter, font)
            else:
                writer.writeUShort(0)
        elif signed:
            writer.writeShort(value)
        else:
            writer.writeUShort(value)