import sys
import io
import copy
import array
import itertools
import struct
import zlib
from collections import namedtuple
from io import BytesIO
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
class _ZoomLevels(list):
    bbiResIncrement = 4
    bbiMaxZoomLevels = 10
    size = _ZoomLevel.formatter.size * bbiMaxZoomLevels

    def __init__(self):
        self[:] = [_ZoomLevel() for i in range(_ZoomLevels.bbiMaxZoomLevels)]

    def __bytes__(self):
        data = b''.join((bytes(item) for item in self))
        data += bytes(_ZoomLevels.size - len(data))
        return data

    @classmethod
    def calculate_reductions(cls, aveSize):
        bbiMaxZoomLevels = _ZoomLevels.bbiMaxZoomLevels
        reductions = np.zeros(bbiMaxZoomLevels, dtype=[('scale', '=i4'), ('size', '=i4'), ('end', '=i4')])
        minZoom = 10
        res = max(int(aveSize), minZoom)
        maxInt = np.iinfo(reductions.dtype['scale']).max
        for resTry in range(bbiMaxZoomLevels):
            if res > maxInt:
                break
            reductions[resTry]['scale'] = res
            res *= _ZoomLevels.bbiResIncrement
        return reductions[:resTry]

    def reduce(self, summaries, initialReduction, buffer, blockSize, itemsPerSlot):
        zoomCount = initialReduction['size']
        reduction = initialReduction['scale'] * _ZoomLevels.bbiResIncrement
        output = buffer.output
        formatter = _RTreeFormatter()
        for zoomLevels in range(1, _ZoomLevels.bbiMaxZoomLevels):
            rezoomCount = len(summaries)
            if rezoomCount >= zoomCount:
                break
            zoomCount = rezoomCount
            self[zoomLevels].dataOffset = output.tell()
            data = zoomCount.to_bytes(4, sys.byteorder)
            output.write(data)
            for summary in summaries:
                buffer.write(summary)
            buffer.flush()
            indexOffset = output.tell()
            formatter.write(summaries, blockSize, itemsPerSlot, indexOffset, output)
            self[zoomLevels].indexOffset = indexOffset
            self[zoomLevels].amount = reduction
            reduction *= _ZoomLevels.bbiResIncrement
            i = 0
            chromId = None
            for summary in summaries:
                if summary.chromId != chromId or summary.end > end:
                    end = summary.start + reduction
                    chromId = summary.chromId
                    currentSummary = summary
                    summaries[i] = currentSummary
                    i += 1
                else:
                    currentSummary += summary
            del summaries[i:]
        del self[zoomLevels:]