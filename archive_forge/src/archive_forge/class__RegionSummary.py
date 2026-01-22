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
class _RegionSummary(_Summary):
    __slots__ = _Region.__slots__ + _Summary.__slots__
    formatter = struct.Struct('=IIIIffff')
    size = formatter.size

    def __init__(self, chromId, start, end, value):
        self.chromId = chromId
        self.start = start
        self.end = end
        self.validCount = 0
        self.minVal = np.float32(value)
        self.maxVal = np.float32(value)
        self.sumData = np.float32(0.0)
        self.sumSquares = np.float32(0.0)
        self.offset = None

    def __iadd__(self, other):
        self.end = other.end
        self.validCount += other.validCount
        self.minVal = min(self.minVal, other.minVal)
        self.maxVal = max(self.maxVal, other.maxVal)
        self.sumData = np.float32(self.sumData + other.sumData)
        self.sumSquares = np.float32(self.sumSquares + other.sumSquares)
        return self

    def update(self, overlap, val):
        self.validCount += overlap
        if self.minVal > val:
            self.minVal = np.float32(val)
        if self.maxVal < val:
            self.maxVal = np.float32(val)
        self.sumData = np.float32(self.sumData + val * overlap)
        self.sumSquares = np.float32(self.sumSquares + val * val * overlap)

    def __bytes__(self):
        return self.formatter.pack(self.chromId, self.start, self.end, self.validCount, self.minVal, self.maxVal, self.sumData, self.sumSquares)