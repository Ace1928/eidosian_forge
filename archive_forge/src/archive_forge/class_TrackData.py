from fontTools.misc import sstruct
from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytesjoin, safeEval
from fontTools.ttLib import TTLibError
from . import DefaultTable
import struct
from collections.abc import MutableMapping
class TrackData(MutableMapping):

    def __init__(self, initialdata={}):
        self._map = dict(initialdata)

    def compile(self, offset):
        nTracks = len(self)
        sizes = self.sizes()
        nSizes = len(sizes)
        offset += TRACK_DATA_FORMAT_SIZE + TRACK_TABLE_ENTRY_FORMAT_SIZE * nTracks
        trackDataHeader = sstruct.pack(TRACK_DATA_FORMAT, {'nTracks': nTracks, 'nSizes': nSizes, 'sizeTableOffset': offset})
        entryDataList = []
        perSizeDataList = []
        offset += SIZE_VALUE_FORMAT_SIZE * nSizes
        for track, entry in sorted(self.items()):
            assert entry.nameIndex is not None
            entry.track = track
            entry.offset = offset
            entryDataList += [sstruct.pack(TRACK_TABLE_ENTRY_FORMAT, entry)]
            for size, value in sorted(entry.items()):
                perSizeDataList += [struct.pack(PER_SIZE_VALUE_FORMAT, value)]
            offset += PER_SIZE_VALUE_FORMAT_SIZE * nSizes
        sizeDataList = [struct.pack(SIZE_VALUE_FORMAT, fl2fi(sv, 16)) for sv in sorted(sizes)]
        data = bytesjoin([trackDataHeader] + entryDataList + sizeDataList + perSizeDataList)
        return data

    def decompile(self, data, offset):
        trackDataHeader = data[offset:offset + TRACK_DATA_FORMAT_SIZE]
        if len(trackDataHeader) != TRACK_DATA_FORMAT_SIZE:
            raise TTLibError('not enough data to decompile TrackData header')
        sstruct.unpack(TRACK_DATA_FORMAT, trackDataHeader, self)
        offset += TRACK_DATA_FORMAT_SIZE
        nSizes = self.nSizes
        sizeTableOffset = self.sizeTableOffset
        sizeTable = []
        for i in range(nSizes):
            sizeValueData = data[sizeTableOffset:sizeTableOffset + SIZE_VALUE_FORMAT_SIZE]
            if len(sizeValueData) < SIZE_VALUE_FORMAT_SIZE:
                raise TTLibError('not enough data to decompile TrackData size subtable')
            sizeValue, = struct.unpack(SIZE_VALUE_FORMAT, sizeValueData)
            sizeTable.append(fi2fl(sizeValue, 16))
            sizeTableOffset += SIZE_VALUE_FORMAT_SIZE
        for i in range(self.nTracks):
            entry = TrackTableEntry()
            entryData = data[offset:offset + TRACK_TABLE_ENTRY_FORMAT_SIZE]
            if len(entryData) < TRACK_TABLE_ENTRY_FORMAT_SIZE:
                raise TTLibError('not enough data to decompile TrackTableEntry record')
            sstruct.unpack(TRACK_TABLE_ENTRY_FORMAT, entryData, entry)
            perSizeOffset = entry.offset
            for j in range(nSizes):
                size = sizeTable[j]
                perSizeValueData = data[perSizeOffset:perSizeOffset + PER_SIZE_VALUE_FORMAT_SIZE]
                if len(perSizeValueData) < PER_SIZE_VALUE_FORMAT_SIZE:
                    raise TTLibError('not enough data to decompile per-size track values')
                perSizeValue, = struct.unpack(PER_SIZE_VALUE_FORMAT, perSizeValueData)
                entry[size] = perSizeValue
                perSizeOffset += PER_SIZE_VALUE_FORMAT_SIZE
            self[entry.track] = entry
            offset += TRACK_TABLE_ENTRY_FORMAT_SIZE

    def toXML(self, writer, ttFont):
        nTracks = len(self)
        nSizes = len(self.sizes())
        writer.comment('nTracks=%d, nSizes=%d' % (nTracks, nSizes))
        writer.newline()
        for track, entry in sorted(self.items()):
            assert entry.nameIndex is not None
            entry.track = track
            entry.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name != 'trackEntry':
            return
        entry = TrackTableEntry()
        entry.fromXML(name, attrs, content, ttFont)
        self[entry.track] = entry

    def sizes(self):
        if not self:
            return frozenset()
        tracks = list(self.tracks())
        sizes = self[tracks.pop(0)].sizes()
        for track in tracks:
            entrySizes = self[track].sizes()
            if sizes != entrySizes:
                raise TTLibError("'trak' table entries must specify the same sizes: %s != %s" % (sorted(sizes), sorted(entrySizes)))
        return frozenset(sizes)

    def __getitem__(self, track):
        return self._map[track]

    def __delitem__(self, track):
        del self._map[track]

    def __setitem__(self, track, entry):
        self._map[track] = entry

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def keys(self):
        return self._map.keys()
    tracks = keys

    def __repr__(self):
        return 'TrackData({})'.format(self._map if self else '')