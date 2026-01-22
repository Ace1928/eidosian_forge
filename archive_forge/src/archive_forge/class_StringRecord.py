from fontTools.misc import sstruct
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
import pdb
import struct
class StringRecord(object):

    def toXML(self, writer, ttFont):
        writer.begintag('StringRecord')
        writer.newline()
        writer.simpletag('labelID', value=self.labelID)
        writer.comment(getLabelString(self.labelID))
        writer.newline()
        writer.newline()
        writer.simpletag('string', value=mapUTF8toXML(self.string))
        writer.newline()
        writer.endtag('StringRecord')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        for element in content:
            if isinstance(element, str):
                continue
            name, attrs, content = element
            value = attrs['value']
            if name == 'string':
                self.string = mapXMLToUTF8(value)
            else:
                setattr(self, name, safeEval(value))

    def compile(self, parentTable):
        data = sstruct.pack(METAStringRecordFormat, self)
        if parentTable.metaFlags == 0:
            datum = struct.pack('>H', self.offset)
        elif parentTable.metaFlags == 1:
            datum = struct.pack('>L', self.offset)
        data = data + datum
        return data

    def __repr__(self):
        return 'StringRecord [ labelID: ' + str(self.labelID) + ' aka ' + getLabelString(self.labelID) + ', offset: ' + str(self.offset) + ', length: ' + str(self.stringLen) + ', string: ' + self.string + ' ]'