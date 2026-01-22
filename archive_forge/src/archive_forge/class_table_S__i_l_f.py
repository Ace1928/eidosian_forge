from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
from . import grUtils
from array import array
from functools import reduce
import struct, re, sys
class table_S__i_l_f(DefaultTable.DefaultTable):
    """Silf table support"""

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.silfs = []

    def decompile(self, data, ttFont):
        sstruct.unpack2(Silf_hdr_format, data, self)
        self.version = float(floatToFixedToStr(self.version, precisionBits=16))
        if self.version >= 5.0:
            data, self.scheme = grUtils.decompress(data)
            sstruct.unpack2(Silf_hdr_format_3, data, self)
            base = sstruct.calcsize(Silf_hdr_format_3)
        elif self.version < 3.0:
            self.numSilf = struct.unpack('>H', data[4:6])
            self.scheme = 0
            self.compilerVersion = 0
            base = 8
        else:
            self.scheme = 0
            sstruct.unpack2(Silf_hdr_format_3, data, self)
            base = sstruct.calcsize(Silf_hdr_format_3)
        silfoffsets = struct.unpack_from('>%dL' % self.numSilf, data[base:])
        for offset in silfoffsets:
            s = Silf()
            self.silfs.append(s)
            s.decompile(data[offset:], ttFont, self.version)

    def compile(self, ttFont):
        self.numSilf = len(self.silfs)
        if self.version < 3.0:
            hdr = sstruct.pack(Silf_hdr_format, self)
            hdr += struct.pack('>HH', self.numSilf, 0)
        else:
            hdr = sstruct.pack(Silf_hdr_format_3, self)
        offset = len(hdr) + 4 * self.numSilf
        data = b''
        for s in self.silfs:
            hdr += struct.pack('>L', offset)
            subdata = s.compile(ttFont, self.version)
            offset += len(subdata)
            data += subdata
        if self.version >= 5.0:
            return grUtils.compress(self.scheme, hdr + data)
        return hdr + data

    def toXML(self, writer, ttFont):
        writer.comment('Attributes starting with _ are informative only')
        writer.newline()
        writer.simpletag('version', version=self.version, compilerVersion=self.compilerVersion, compressionScheme=self.scheme)
        writer.newline()
        for s in self.silfs:
            writer.begintag('silf')
            writer.newline()
            s.toXML(writer, ttFont, self.version)
            writer.endtag('silf')
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'version':
            self.scheme = int(safeEval(attrs['compressionScheme']))
            self.version = float(safeEval(attrs['version']))
            self.compilerVersion = int(safeEval(attrs['compilerVersion']))
            return
        if name == 'silf':
            s = Silf()
            self.silfs.append(s)
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                s.fromXML(tag, attrs, subcontent, ttFont, self.version)