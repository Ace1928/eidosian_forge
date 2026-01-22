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
class _ExtraIndices(list):
    formatter = struct.Struct('=HHQ52x')

    def __init__(self, names, declaration):
        self[:] = [_ExtraIndex(name, declaration) for name in names]

    @property
    def size(self):
        return self.formatter.size + _ExtraIndex.formatter.size * len(self)

    def initialize(self, bedCount):
        if bedCount == 0:
            return
        for extra_index in self:
            keySize = extra_index.maxFieldSize
            dtype = np.dtype([('name', f'=S{keySize}'), ('offset', '=u8'), ('size', '=u8')])
            extra_index.chunks = np.zeros(bedCount, dtype=dtype)

    def tofile(self, stream):
        size = self.formatter.size
        if len(self) > 0:
            offset = stream.tell() + size
            data = self.formatter.pack(size, len(self), offset)
            stream.write(data)
            for extra_index in self:
                stream.write(bytes(extra_index))
        else:
            data = self.formatter.pack(size, 0, 0)
            stream.write(data)