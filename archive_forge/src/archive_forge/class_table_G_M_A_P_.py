from fontTools.misc import sstruct
from fontTools.misc.textTools import tobytes, tostr, safeEval
from . import DefaultTable
class table_G_M_A_P_(DefaultTable.DefaultTable):
    dependencies = []

    def decompile(self, data, ttFont):
        dummy, newData = sstruct.unpack2(GMAPFormat, data, self)
        self.psFontName = tostr(newData[:self.fontNameLength])
        assert self.recordsOffset % 4 == 0, 'GMAP error: recordsOffset is not 32 bit aligned.'
        newData = data[self.recordsOffset:]
        self.gmapRecords = []
        for i in range(self.recordsCount):
            gmapRecord, newData = sstruct.unpack2(GMAPRecordFormat1, newData, GMAPRecord())
            gmapRecord.name = gmapRecord.name.strip('\x00')
            self.gmapRecords.append(gmapRecord)

    def compile(self, ttFont):
        self.recordsCount = len(self.gmapRecords)
        self.fontNameLength = len(self.psFontName)
        self.recordsOffset = 4 * ((self.fontNameLength + 12 + 3) // 4)
        data = sstruct.pack(GMAPFormat, self)
        data = data + tobytes(self.psFontName)
        data = data + b'\x00' * (self.recordsOffset - len(data))
        for record in self.gmapRecords:
            data = data + record.compile(ttFont)
        return data

    def toXML(self, writer, ttFont):
        writer.comment('Most of this table will be recalculated by the compiler')
        writer.newline()
        formatstring, names, fixes = sstruct.getformat(GMAPFormat)
        for name in names:
            value = getattr(self, name)
            writer.simpletag(name, value=value)
            writer.newline()
        writer.simpletag('PSFontName', value=self.psFontName)
        writer.newline()
        for gmapRecord in self.gmapRecords:
            gmapRecord.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'GMAPRecord':
            if not hasattr(self, 'gmapRecords'):
                self.gmapRecords = []
            gmapRecord = GMAPRecord()
            self.gmapRecords.append(gmapRecord)
            for element in content:
                if isinstance(element, str):
                    continue
                name, attrs, content = element
                gmapRecord.fromXML(name, attrs, content, ttFont)
        else:
            value = attrs['value']
            if name == 'PSFontName':
                self.psFontName = value
            else:
                setattr(self, name, safeEval(value))