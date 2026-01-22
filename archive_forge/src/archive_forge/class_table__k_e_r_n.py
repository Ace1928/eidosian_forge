from fontTools.ttLib import getSearchRange
from fontTools.misc.textTools import safeEval, readHex
from fontTools.misc.fixedTools import fixedToFloat as fi2fl, floatToFixed as fl2fi
from . import DefaultTable
import struct
import sys
import array
import logging
class table__k_e_r_n(DefaultTable.DefaultTable):

    def getkern(self, format):
        for subtable in self.kernTables:
            if subtable.format == format:
                return subtable
        return None

    def decompile(self, data, ttFont):
        version, nTables = struct.unpack('>HH', data[:4])
        apple = False
        if len(data) >= 8 and version == 1:
            version, nTables = struct.unpack('>LL', data[:8])
            self.version = fi2fl(version, 16)
            data = data[8:]
            apple = True
        else:
            self.version = version
            data = data[4:]
        self.kernTables = []
        for i in range(nTables):
            if self.version == 1.0:
                length, coverage, subtableFormat = struct.unpack('>LBB', data[:6])
            else:
                _, length, subtableFormat, coverage = struct.unpack('>HHBB', data[:6])
                if nTables == 1 and subtableFormat == 0:
                    nPairs, = struct.unpack('>H', data[6:8])
                    calculated_length = nPairs * 6 + 14
                    if length != calculated_length:
                        log.warning("'kern' subtable longer than defined: %d bytes instead of %d bytes" % (calculated_length, length))
                    length = calculated_length
            if subtableFormat not in kern_classes:
                subtable = KernTable_format_unkown(subtableFormat)
            else:
                subtable = kern_classes[subtableFormat](apple)
            subtable.decompile(data[:length], ttFont)
            self.kernTables.append(subtable)
            data = data[length:]

    def compile(self, ttFont):
        if hasattr(self, 'kernTables'):
            nTables = len(self.kernTables)
        else:
            nTables = 0
        if self.version == 1.0:
            data = struct.pack('>LL', fl2fi(self.version, 16), nTables)
        else:
            data = struct.pack('>HH', self.version, nTables)
        if hasattr(self, 'kernTables'):
            for subtable in self.kernTables:
                data = data + subtable.compile(ttFont)
        return data

    def toXML(self, writer, ttFont):
        writer.simpletag('version', value=self.version)
        writer.newline()
        for subtable in self.kernTables:
            subtable.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'version':
            self.version = safeEval(attrs['value'])
            return
        if name != 'kernsubtable':
            return
        if not hasattr(self, 'kernTables'):
            self.kernTables = []
        format = safeEval(attrs['format'])
        if format not in kern_classes:
            subtable = KernTable_format_unkown(format)
        else:
            apple = self.version == 1.0
            subtable = kern_classes[format](apple)
        self.kernTables.append(subtable)
        subtable.fromXML(name, attrs, content, ttFont)