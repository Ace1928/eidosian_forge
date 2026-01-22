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
class OverflowErrorRecord(object):

    def __init__(self, overflowTuple):
        self.tableType = overflowTuple[0]
        self.LookupListIndex = overflowTuple[1]
        self.SubTableIndex = overflowTuple[2]
        self.itemName = overflowTuple[3]
        self.itemIndex = overflowTuple[4]

    def __repr__(self):
        return str((self.tableType, 'LookupIndex:', self.LookupListIndex, 'SubTableIndex:', self.SubTableIndex, 'ItemName:', self.itemName, 'ItemIndex:', self.itemIndex))