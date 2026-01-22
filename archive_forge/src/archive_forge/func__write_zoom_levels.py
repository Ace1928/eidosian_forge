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
def _write_zoom_levels(self, alignments, output, dataSize, chromUsageList, reductions):
    zoomList = _ZoomLevels()
    totalSum = _Summary()
    if len(alignments) == 0:
        totalSum.minVal = 0.0
        totalSum.maxVal = 0.0
    else:
        blockSize = self.blockSize
        doCompress = self.compress
        itemsPerSlot = self.itemsPerSlot
        maxReducedSize = dataSize / 2
        zoomList[0].dataOffset = output.tell()
        for initialReduction in reductions:
            reducedSize = initialReduction['size'] * _RegionSummary.size
            if doCompress:
                reducedSize /= 2
            if reducedSize <= maxReducedSize:
                break
        else:
            initialReduction = reductions[0]
        initialReduction['size'].tofile(output)
        size = itemsPerSlot * _RegionSummary.size
        if doCompress:
            buffer = _ZippedBufferedStream(output, size)
        else:
            buffer = _BufferedStream(output, size)
        regions = []
        rezoomedList = []
        trees = _RangeTree.generate(chromUsageList, alignments)
        scale = initialReduction['scale']
        doubleReductionSize = scale * _ZoomLevels.bbiResIncrement
        for tree in trees:
            start = -sys.maxsize
            summaries = tree.generate_summaries(scale, totalSum)
            for summary in summaries:
                buffer.write(summary)
                regions.append(summary)
                if start + doubleReductionSize < summary.end:
                    rezoomed = copy.copy(summary)
                    start = rezoomed.start
                    rezoomedList.append(rezoomed)
                else:
                    rezoomed += summary
        buffer.flush()
        assert len(regions) == initialReduction['size']
        zoomList[0].amount = initialReduction['scale']
        indexOffset = output.tell()
        zoomList[0].indexOffset = indexOffset
        _RTreeFormatter().write(regions, blockSize, itemsPerSlot, indexOffset, output)
        if doCompress:
            buffer = _ZippedBufferedStream(output, size)
        else:
            buffer = _BufferedStream(output, _RegionSummary.size)
        zoomList.reduce(rezoomedList, initialReduction, buffer, blockSize, itemsPerSlot)
    return (zoomList, totalSum)