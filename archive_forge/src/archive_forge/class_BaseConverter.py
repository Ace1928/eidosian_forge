from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
class BaseConverter(object):
    """Base class for converter objects. Apart from the constructor, this
    is an abstract class."""

    def __init__(self, name, repeat, aux, tableClass=None, *, description=''):
        self.name = name
        self.repeat = repeat
        self.aux = aux
        self.tableClass = tableClass
        self.isCount = name.endswith('Count') or name in ['DesignAxisRecordSize', 'ValueRecordSize']
        self.isLookupType = name.endswith('LookupType') or name == 'MorphType'
        self.isPropagated = name in ['ClassCount', 'Class2Count', 'FeatureTag', 'SettingsCount', 'VarRegionCount', 'MappingCount', 'RegionAxisCount', 'DesignAxisCount', 'DesignAxisRecordSize', 'AxisValueCount', 'ValueRecordSize', 'AxisCount', 'BaseGlyphRecordCount', 'LayerRecordCount']
        self.description = description

    def readArray(self, reader, font, tableDict, count):
        """Read an array of values from the reader."""
        lazy = font.lazy and count > 8
        if lazy:
            recordSize = self.getRecordSize(reader)
            if recordSize is NotImplemented:
                lazy = False
        if not lazy:
            l = []
            for i in range(count):
                l.append(self.read(reader, font, tableDict))
            return l
        else:
            l = _LazyList()
            l.reader = reader.copy()
            l.pos = l.reader.pos
            l.font = font
            l.conv = self
            l.recordSize = recordSize
            l.extend((_MissingItem([i]) for i in range(count)))
            reader.advance(count * recordSize)
            return l

    def getRecordSize(self, reader):
        if hasattr(self, 'staticSize'):
            return self.staticSize
        return NotImplemented

    def read(self, reader, font, tableDict):
        """Read a value from the reader."""
        raise NotImplementedError(self)

    def writeArray(self, writer, font, tableDict, values):
        try:
            for i, value in enumerate(values):
                self.write(writer, font, tableDict, value, i)
        except Exception as e:
            e.args = e.args + (i,)
            raise

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        """Write a value to the writer."""
        raise NotImplementedError(self)

    def xmlRead(self, attrs, content, font):
        """Read a value from XML."""
        raise NotImplementedError(self)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        """Write a value to XML."""
        raise NotImplementedError(self)
    varIndexBasePlusOffsetRE = re.compile('VarIndexBase\\s*\\+\\s*(\\d+)')

    def getVarIndexOffset(self) -> Optional[int]:
        """If description has `VarIndexBase + {offset}`, return the offset else None."""
        m = self.varIndexBasePlusOffsetRE.search(self.description)
        if not m:
            return None
        return int(m.group(1))