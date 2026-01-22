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
class FormatSwitchingBaseTable(BaseTable):
    """Minor specialization of BaseTable, for tables that have multiple
    formats, eg. CoverageFormat1 vs. CoverageFormat2."""

    @classmethod
    def getRecordSize(cls, reader):
        return NotImplemented

    def getConverters(self):
        try:
            fmt = self.Format
        except AttributeError:
            return []
        return self.converters.get(self.Format, [])

    def getConverterByName(self, name):
        return self.convertersByName[self.Format][name]

    def readFormat(self, reader):
        self.Format = reader.readUShort()

    def writeFormat(self, writer):
        writer.writeUShort(self.Format)

    def toXML(self, xmlWriter, font, attrs=None, name=None):
        BaseTable.toXML(self, xmlWriter, font, attrs, name)

    def getVariableAttrs(self):
        return getVariableAttrs(self.__class__, self.Format)