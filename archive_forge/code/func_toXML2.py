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
def toXML2(self, xmlWriter, font):
    for conv in self.getConverters():
        if conv.repeat:
            value = getattr(self, conv.name, [])
            for i in range(len(value)):
                item = value[i]
                conv.xmlWrite(xmlWriter, font, item, conv.name, [('index', i)])
        else:
            if conv.aux and (not eval(conv.aux, None, vars(self))):
                continue
            value = getattr(self, conv.name, None)
            conv.xmlWrite(xmlWriter, font, value, conv.name, [])