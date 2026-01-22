from fontTools.misc import sstruct
from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import Tag, bytesjoin, safeEval
from fontTools.ttLib import TTLibError
from . import DefaultTable
import struct
class NamedInstance(object):

    def __init__(self):
        self.subfamilyNameID = 0
        self.postscriptNameID = 65535
        self.flags = 0
        self.coordinates = {}

    def compile(self, axisTags, includePostScriptName):
        result = [sstruct.pack(FVAR_INSTANCE_FORMAT, self)]
        for axis in axisTags:
            fixedCoord = fl2fi(self.coordinates[axis], 16)
            result.append(struct.pack('>l', fixedCoord))
        if includePostScriptName:
            result.append(struct.pack('>H', self.postscriptNameID))
        return bytesjoin(result)

    def decompile(self, data, axisTags):
        sstruct.unpack2(FVAR_INSTANCE_FORMAT, data, self)
        pos = sstruct.calcsize(FVAR_INSTANCE_FORMAT)
        for axis in axisTags:
            value = struct.unpack('>l', data[pos:pos + 4])[0]
            self.coordinates[axis] = fi2fl(value, 16)
            pos += 4
        if pos + 2 <= len(data):
            self.postscriptNameID = struct.unpack('>H', data[pos:pos + 2])[0]
        else:
            self.postscriptNameID = 65535

    def toXML(self, writer, ttFont):
        name = ttFont['name'].getDebugName(self.subfamilyNameID) if 'name' in ttFont else None
        if name is not None:
            writer.newline()
            writer.comment(name)
            writer.newline()
        psname = ttFont['name'].getDebugName(self.postscriptNameID) if 'name' in ttFont else None
        if psname is not None:
            writer.comment('PostScript: ' + psname)
            writer.newline()
        if self.postscriptNameID == 65535:
            writer.begintag('NamedInstance', flags='0x%X' % self.flags, subfamilyNameID=self.subfamilyNameID)
        else:
            writer.begintag('NamedInstance', flags='0x%X' % self.flags, subfamilyNameID=self.subfamilyNameID, postscriptNameID=self.postscriptNameID)
        writer.newline()
        for axis in ttFont['fvar'].axes:
            writer.simpletag('coord', axis=axis.axisTag, value=fl2str(self.coordinates[axis.axisTag], 16))
            writer.newline()
        writer.endtag('NamedInstance')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        assert name == 'NamedInstance'
        self.subfamilyNameID = safeEval(attrs['subfamilyNameID'])
        self.flags = safeEval(attrs.get('flags', '0'))
        if 'postscriptNameID' in attrs:
            self.postscriptNameID = safeEval(attrs['postscriptNameID'])
        else:
            self.postscriptNameID = 65535
        for tag, elementAttrs, _ in filter(lambda t: type(t) is tuple, content):
            if tag == 'coord':
                value = str2fl(elementAttrs['value'], 16)
                self.coordinates[elementAttrs['axis']] = value