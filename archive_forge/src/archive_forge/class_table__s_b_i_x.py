from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from . import DefaultTable
from .sbixStrike import Strike
class table__s_b_i_x(DefaultTable.DefaultTable):

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.version = 1
        self.flags = 1
        self.numStrikes = 0
        self.strikes = {}
        self.strikeOffsets = []

    def decompile(self, data, ttFont):
        sstruct.unpack(sbixHeaderFormat, data[:sbixHeaderFormatSize], self)
        for i in range(self.numStrikes):
            current_offset = sbixHeaderFormatSize + i * sbixStrikeOffsetFormatSize
            offset_entry = sbixStrikeOffset()
            sstruct.unpack(sbixStrikeOffsetFormat, data[current_offset:current_offset + sbixStrikeOffsetFormatSize], offset_entry)
            self.strikeOffsets.append(offset_entry.strikeOffset)
        for i in range(self.numStrikes - 1, -1, -1):
            current_strike = Strike(rawdata=data[self.strikeOffsets[i]:])
            data = data[:self.strikeOffsets[i]]
            current_strike.decompile(ttFont)
            if current_strike.ppem in self.strikes:
                from fontTools import ttLib
                raise ttLib.TTLibError("Pixel 'ppem' must be unique for each Strike")
            self.strikes[current_strike.ppem] = current_strike
        del self.strikeOffsets
        del self.numStrikes

    def compile(self, ttFont):
        sbixData = b''
        self.numStrikes = len(self.strikes)
        sbixHeader = sstruct.pack(sbixHeaderFormat, self)
        setOffset = sbixHeaderFormatSize + sbixStrikeOffsetFormatSize * self.numStrikes
        for si in sorted(self.strikes.keys()):
            current_strike = self.strikes[si]
            current_strike.compile(ttFont)
            current_strike.strikeOffset = setOffset
            sbixHeader += sstruct.pack(sbixStrikeOffsetFormat, current_strike)
            setOffset += len(current_strike.data)
            sbixData += current_strike.data
        return sbixHeader + sbixData

    def toXML(self, xmlWriter, ttFont):
        xmlWriter.simpletag('version', value=self.version)
        xmlWriter.newline()
        xmlWriter.simpletag('flags', value=num2binary(self.flags, 16))
        xmlWriter.newline()
        for i in sorted(self.strikes.keys()):
            self.strikes[i].toXML(xmlWriter, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'version':
            setattr(self, name, safeEval(attrs['value']))
        elif name == 'flags':
            setattr(self, name, binary2num(attrs['value']))
        elif name == 'strike':
            current_strike = Strike()
            for element in content:
                if isinstance(element, tuple):
                    name, attrs, content = element
                    current_strike.fromXML(name, attrs, content, ttFont)
            self.strikes[current_strike.ppem] = current_strike
        else:
            from fontTools import ttLib
            raise ttLib.TTLibError("can't handle '%s' element" % name)