from fontTools.misc import sstruct
from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytesjoin, safeEval
from fontTools.ttLib import TTLibError
from . import DefaultTable
import struct
from collections.abc import MutableMapping
class table__t_r_a_k(DefaultTable.DefaultTable):
    dependencies = ['name']

    def compile(self, ttFont):
        dataList = []
        offset = TRAK_HEADER_FORMAT_SIZE
        for direction in ('horiz', 'vert'):
            trackData = getattr(self, direction + 'Data', TrackData())
            offsetName = direction + 'Offset'
            if not trackData:
                setattr(self, offsetName, 0)
                continue
            alignedOffset = offset + 3 & ~3
            padding, offset = (b'\x00' * (alignedOffset - offset), alignedOffset)
            setattr(self, offsetName, offset)
            data = trackData.compile(offset)
            offset += len(data)
            dataList.append(padding + data)
        self.reserved = 0
        tableData = bytesjoin([sstruct.pack(TRAK_HEADER_FORMAT, self)] + dataList)
        return tableData

    def decompile(self, data, ttFont):
        sstruct.unpack(TRAK_HEADER_FORMAT, data[:TRAK_HEADER_FORMAT_SIZE], self)
        for direction in ('horiz', 'vert'):
            trackData = TrackData()
            offset = getattr(self, direction + 'Offset')
            if offset != 0:
                trackData.decompile(data, offset)
            setattr(self, direction + 'Data', trackData)

    def toXML(self, writer, ttFont):
        writer.simpletag('version', value=self.version)
        writer.newline()
        writer.simpletag('format', value=self.format)
        writer.newline()
        for direction in ('horiz', 'vert'):
            dataName = direction + 'Data'
            writer.begintag(dataName)
            writer.newline()
            trackData = getattr(self, dataName, TrackData())
            trackData.toXML(writer, ttFont)
            writer.endtag(dataName)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'version':
            self.version = safeEval(attrs['value'])
        elif name == 'format':
            self.format = safeEval(attrs['value'])
        elif name in ('horizData', 'vertData'):
            trackData = TrackData()
            setattr(self, name, trackData)
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content_ = element
                trackData.fromXML(name, attrs, content_, ttFont)