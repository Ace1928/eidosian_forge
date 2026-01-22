from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
class table_O_S_2f_2(DefaultTable.DefaultTable):
    """the OS/2 table"""
    dependencies = ['head']

    def decompile(self, data, ttFont):
        dummy, data = sstruct.unpack2(OS2_format_0, data, self)
        if self.version == 1:
            dummy, data = sstruct.unpack2(OS2_format_1_addition, data, self)
        elif self.version in (2, 3, 4):
            dummy, data = sstruct.unpack2(OS2_format_2_addition, data, self)
        elif self.version == 5:
            dummy, data = sstruct.unpack2(OS2_format_5_addition, data, self)
            self.usLowerOpticalPointSize /= 20
            self.usUpperOpticalPointSize /= 20
        elif self.version != 0:
            from fontTools import ttLib
            raise ttLib.TTLibError('unknown format for OS/2 table: version %s' % self.version)
        if len(data):
            log.warning("too much 'OS/2' table data")
        self.panose = sstruct.unpack(panoseFormat, self.panose, Panose())

    def compile(self, ttFont):
        self.updateFirstAndLastCharIndex(ttFont)
        panose = self.panose
        head = ttFont['head']
        if self.fsSelection & 1 and (not head.macStyle & 1 << 1):
            log.warning('fsSelection bit 0 (italic) and head table macStyle bit 1 (italic) should match')
        if self.fsSelection & 1 << 5 and (not head.macStyle & 1):
            log.warning('fsSelection bit 5 (bold) and head table macStyle bit 0 (bold) should match')
        if self.fsSelection & 1 << 6 and self.fsSelection & 1 + (1 << 5):
            log.warning('fsSelection bit 6 (regular) is set, bits 0 (italic) and 5 (bold) must be clear')
        if self.version < 4 and self.fsSelection & 896:
            log.warning('fsSelection bits 7, 8 and 9 are only defined in OS/2 table version 4 and up: version %s', self.version)
        self.panose = sstruct.pack(panoseFormat, self.panose)
        if self.version == 0:
            data = sstruct.pack(OS2_format_0, self)
        elif self.version == 1:
            data = sstruct.pack(OS2_format_1, self)
        elif self.version in (2, 3, 4):
            data = sstruct.pack(OS2_format_2, self)
        elif self.version == 5:
            d = self.__dict__.copy()
            d['usLowerOpticalPointSize'] = round(self.usLowerOpticalPointSize * 20)
            d['usUpperOpticalPointSize'] = round(self.usUpperOpticalPointSize * 20)
            data = sstruct.pack(OS2_format_5, d)
        else:
            from fontTools import ttLib
            raise ttLib.TTLibError('unknown format for OS/2 table: version %s' % self.version)
        self.panose = panose
        return data

    def toXML(self, writer, ttFont):
        writer.comment("The fields 'usFirstCharIndex' and 'usLastCharIndex'\nwill be recalculated by the compiler")
        writer.newline()
        if self.version == 1:
            format = OS2_format_1
        elif self.version in (2, 3, 4):
            format = OS2_format_2
        elif self.version == 5:
            format = OS2_format_5
        else:
            format = OS2_format_0
        formatstring, names, fixes = sstruct.getformat(format)
        for name in names:
            value = getattr(self, name)
            if name == 'panose':
                writer.begintag('panose')
                writer.newline()
                value.toXML(writer, ttFont)
                writer.endtag('panose')
            elif name in ('ulUnicodeRange1', 'ulUnicodeRange2', 'ulUnicodeRange3', 'ulUnicodeRange4', 'ulCodePageRange1', 'ulCodePageRange2'):
                writer.simpletag(name, value=num2binary(value))
            elif name in ('fsType', 'fsSelection'):
                writer.simpletag(name, value=num2binary(value, 16))
            elif name == 'achVendID':
                writer.simpletag(name, value=repr(value)[1:-1])
            else:
                writer.simpletag(name, value=value)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'panose':
            self.panose = panose = Panose()
            for element in content:
                if isinstance(element, tuple):
                    name, attrs, content = element
                    panose.fromXML(name, attrs, content, ttFont)
        elif name in ('ulUnicodeRange1', 'ulUnicodeRange2', 'ulUnicodeRange3', 'ulUnicodeRange4', 'ulCodePageRange1', 'ulCodePageRange2', 'fsType', 'fsSelection'):
            setattr(self, name, binary2num(attrs['value']))
        elif name == 'achVendID':
            setattr(self, name, safeEval("'''" + attrs['value'] + "'''"))
        else:
            setattr(self, name, safeEval(attrs['value']))

    def updateFirstAndLastCharIndex(self, ttFont):
        if 'cmap' not in ttFont:
            return
        codes = set()
        for table in getattr(ttFont['cmap'], 'tables', []):
            if table.isUnicode():
                codes.update(table.cmap.keys())
        if codes:
            minCode = min(codes)
            maxCode = max(codes)
            self.usFirstCharIndex = min(65535, minCode)
            self.usLastCharIndex = min(65535, maxCode)

    @property
    def usMaxContex(self):
        return self.usMaxContext

    @usMaxContex.setter
    def usMaxContex(self, value):
        self.usMaxContext = value

    @property
    def fsFirstCharIndex(self):
        return self.usFirstCharIndex

    @fsFirstCharIndex.setter
    def fsFirstCharIndex(self, value):
        self.usFirstCharIndex = value

    @property
    def fsLastCharIndex(self):
        return self.usLastCharIndex

    @fsLastCharIndex.setter
    def fsLastCharIndex(self, value):
        self.usLastCharIndex = value

    def getUnicodeRanges(self):
        """Return the set of 'ulUnicodeRange*' bits currently enabled."""
        bits = set()
        ul1, ul2 = (self.ulUnicodeRange1, self.ulUnicodeRange2)
        ul3, ul4 = (self.ulUnicodeRange3, self.ulUnicodeRange4)
        for i in range(32):
            if ul1 & 1 << i:
                bits.add(i)
            if ul2 & 1 << i:
                bits.add(i + 32)
            if ul3 & 1 << i:
                bits.add(i + 64)
            if ul4 & 1 << i:
                bits.add(i + 96)
        return bits

    def setUnicodeRanges(self, bits):
        """Set the 'ulUnicodeRange*' fields to the specified 'bits'."""
        ul1, ul2, ul3, ul4 = (0, 0, 0, 0)
        for bit in bits:
            if 0 <= bit < 32:
                ul1 |= 1 << bit
            elif 32 <= bit < 64:
                ul2 |= 1 << bit - 32
            elif 64 <= bit < 96:
                ul3 |= 1 << bit - 64
            elif 96 <= bit < 123:
                ul4 |= 1 << bit - 96
            else:
                raise ValueError('expected 0 <= int <= 122, found: %r' % bit)
        self.ulUnicodeRange1, self.ulUnicodeRange2 = (ul1, ul2)
        self.ulUnicodeRange3, self.ulUnicodeRange4 = (ul3, ul4)

    def recalcUnicodeRanges(self, ttFont, pruneOnly=False):
        """Intersect the codepoints in the font's Unicode cmap subtables with
        the Unicode block ranges defined in the OpenType specification (v1.7),
        and set the respective 'ulUnicodeRange*' bits if there is at least ONE
        intersection.
        If 'pruneOnly' is True, only clear unused bits with NO intersection.
        """
        unicodes = set()
        for table in ttFont['cmap'].tables:
            if table.isUnicode():
                unicodes.update(table.cmap.keys())
        if pruneOnly:
            empty = intersectUnicodeRanges(unicodes, inverse=True)
            bits = self.getUnicodeRanges() - empty
        else:
            bits = intersectUnicodeRanges(unicodes)
        self.setUnicodeRanges(bits)
        return bits

    def getCodePageRanges(self):
        """Return the set of 'ulCodePageRange*' bits currently enabled."""
        bits = set()
        if self.version < 1:
            return bits
        ul1, ul2 = (self.ulCodePageRange1, self.ulCodePageRange2)
        for i in range(32):
            if ul1 & 1 << i:
                bits.add(i)
            if ul2 & 1 << i:
                bits.add(i + 32)
        return bits

    def setCodePageRanges(self, bits):
        """Set the 'ulCodePageRange*' fields to the specified 'bits'."""
        ul1, ul2 = (0, 0)
        for bit in bits:
            if 0 <= bit < 32:
                ul1 |= 1 << bit
            elif 32 <= bit < 64:
                ul2 |= 1 << bit - 32
            else:
                raise ValueError(f'expected 0 <= int <= 63, found: {bit:r}')
        if self.version < 1:
            self.version = 1
        self.ulCodePageRange1, self.ulCodePageRange2 = (ul1, ul2)

    def recalcCodePageRanges(self, ttFont, pruneOnly=False):
        unicodes = set()
        for table in ttFont['cmap'].tables:
            if table.isUnicode():
                unicodes.update(table.cmap.keys())
        bits = calcCodePageRanges(unicodes)
        if pruneOnly:
            bits &= self.getCodePageRanges()
        if not bits:
            bits = {0}
        self.setCodePageRanges(bits)
        return bits

    def recalcAvgCharWidth(self, ttFont):
        """Recalculate xAvgCharWidth using metrics from ttFont's 'hmtx' table.

        Set it to 0 if the unlikely event 'hmtx' table is not found.
        """
        avg_width = 0
        hmtx = ttFont.get('hmtx')
        if hmtx is not None:
            widths = [width for width, _ in hmtx.metrics.values() if width > 0]
            if widths:
                avg_width = otRound(sum(widths) / len(widths))
        self.xAvgCharWidth = avg_width
        return avg_width