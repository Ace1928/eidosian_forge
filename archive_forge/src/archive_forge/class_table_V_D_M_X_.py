from . import DefaultTable
from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
import struct
class table_V_D_M_X_(DefaultTable.DefaultTable):

    def decompile(self, data, ttFont):
        pos = 0
        dummy, data = sstruct.unpack2(VDMX_HeaderFmt, data, self)
        pos += sstruct.calcsize(VDMX_HeaderFmt)
        self.ratRanges = []
        for i in range(self.numRatios):
            ratio, data = sstruct.unpack2(VDMX_RatRangeFmt, data)
            pos += sstruct.calcsize(VDMX_RatRangeFmt)
            ratio['groupIndex'] = None
            self.ratRanges.append(ratio)
        lenOffset = struct.calcsize('>H')
        _offsets = []
        for i in range(self.numRatios):
            offset = struct.unpack('>H', data[0:lenOffset])[0]
            data = data[lenOffset:]
            pos += lenOffset
            _offsets.append(offset)
        self.groups = []
        for groupIndex in range(self.numRecs):
            currOffset = pos
            group, data = sstruct.unpack2(VDMX_GroupFmt, data)
            recs = group.pop('recs')
            startsz = group.pop('startsz')
            endsz = group.pop('endsz')
            pos += sstruct.calcsize(VDMX_GroupFmt)
            for j in range(recs):
                vTable, data = sstruct.unpack2(VDMX_vTableFmt, data)
                vTableLength = sstruct.calcsize(VDMX_vTableFmt)
                pos += vTableLength
                group[vTable['yPelHeight']] = (vTable['yMax'], vTable['yMin'])
            minSize = min(group.keys())
            maxSize = max(group.keys())
            assert startsz == minSize, 'startsz (%s) must equal min yPelHeight (%s): group %d' % (group.startsz, minSize, groupIndex)
            assert endsz == maxSize, 'endsz (%s) must equal max yPelHeight (%s): group %d' % (group.endsz, maxSize, groupIndex)
            self.groups.append(group)
            for offsetIndex, offsetValue in enumerate(_offsets):
                if currOffset == offsetValue:
                    self.ratRanges[offsetIndex]['groupIndex'] = groupIndex
        for i in range(self.numRatios):
            ratio = self.ratRanges[i]
            if ratio['groupIndex'] is None:
                from fontTools import ttLib
                raise ttLib.TTLibError('no group defined for ratRange %d' % i)

    def _getOffsets(self):
        """
        Calculate offsets to VDMX_Group records.
        For each ratRange return a list of offset values from the beginning of
        the VDMX table to a VDMX_Group.
        """
        lenHeader = sstruct.calcsize(VDMX_HeaderFmt)
        lenRatRange = sstruct.calcsize(VDMX_RatRangeFmt)
        lenOffset = struct.calcsize('>H')
        lenGroupHeader = sstruct.calcsize(VDMX_GroupFmt)
        lenVTable = sstruct.calcsize(VDMX_vTableFmt)
        pos = lenHeader + self.numRatios * lenRatRange + self.numRatios * lenOffset
        groupOffsets = []
        for group in self.groups:
            groupOffsets.append(pos)
            lenGroup = lenGroupHeader + len(group) * lenVTable
            pos += lenGroup
        offsets = []
        for ratio in self.ratRanges:
            groupIndex = ratio['groupIndex']
            offsets.append(groupOffsets[groupIndex])
        return offsets

    def compile(self, ttFont):
        if not (self.version == 0 or self.version == 1):
            from fontTools import ttLib
            raise ttLib.TTLibError('unknown format for VDMX table: version %s' % self.version)
        data = sstruct.pack(VDMX_HeaderFmt, self)
        for ratio in self.ratRanges:
            data += sstruct.pack(VDMX_RatRangeFmt, ratio)
        for offset in self._getOffsets():
            data += struct.pack('>H', offset)
        for group in self.groups:
            recs = len(group)
            startsz = min(group.keys())
            endsz = max(group.keys())
            gHeader = {'recs': recs, 'startsz': startsz, 'endsz': endsz}
            data += sstruct.pack(VDMX_GroupFmt, gHeader)
            for yPelHeight, (yMax, yMin) in sorted(group.items()):
                vTable = {'yPelHeight': yPelHeight, 'yMax': yMax, 'yMin': yMin}
                data += sstruct.pack(VDMX_vTableFmt, vTable)
        return data

    def toXML(self, writer, ttFont):
        writer.simpletag('version', value=self.version)
        writer.newline()
        writer.begintag('ratRanges')
        writer.newline()
        for ratio in self.ratRanges:
            groupIndex = ratio['groupIndex']
            writer.simpletag('ratRange', bCharSet=ratio['bCharSet'], xRatio=ratio['xRatio'], yStartRatio=ratio['yStartRatio'], yEndRatio=ratio['yEndRatio'], groupIndex=groupIndex)
            writer.newline()
        writer.endtag('ratRanges')
        writer.newline()
        writer.begintag('groups')
        writer.newline()
        for groupIndex in range(self.numRecs):
            group = self.groups[groupIndex]
            recs = len(group)
            startsz = min(group.keys())
            endsz = max(group.keys())
            writer.begintag('group', index=groupIndex)
            writer.newline()
            writer.comment('recs=%d, startsz=%d, endsz=%d' % (recs, startsz, endsz))
            writer.newline()
            for yPelHeight, (yMax, yMin) in sorted(group.items()):
                writer.simpletag('record', [('yPelHeight', yPelHeight), ('yMax', yMax), ('yMin', yMin)])
                writer.newline()
            writer.endtag('group')
            writer.newline()
        writer.endtag('groups')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'version':
            self.version = safeEval(attrs['value'])
        elif name == 'ratRanges':
            if not hasattr(self, 'ratRanges'):
                self.ratRanges = []
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name == 'ratRange':
                    if not hasattr(self, 'numRatios'):
                        self.numRatios = 1
                    else:
                        self.numRatios += 1
                    ratio = {'bCharSet': safeEval(attrs['bCharSet']), 'xRatio': safeEval(attrs['xRatio']), 'yStartRatio': safeEval(attrs['yStartRatio']), 'yEndRatio': safeEval(attrs['yEndRatio']), 'groupIndex': safeEval(attrs['groupIndex'])}
                    self.ratRanges.append(ratio)
        elif name == 'groups':
            if not hasattr(self, 'groups'):
                self.groups = []
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name == 'group':
                    if not hasattr(self, 'numRecs'):
                        self.numRecs = 1
                    else:
                        self.numRecs += 1
                    group = {}
                    for element in content:
                        if not isinstance(element, tuple):
                            continue
                        name, attrs, content = element
                        if name == 'record':
                            yPelHeight = safeEval(attrs['yPelHeight'])
                            yMax = safeEval(attrs['yMax'])
                            yMin = safeEval(attrs['yMin'])
                            group[yPelHeight] = (yMax, yMin)
                    self.groups.append(group)