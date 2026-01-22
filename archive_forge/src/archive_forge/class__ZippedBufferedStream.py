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
class _ZippedBufferedStream(_BufferedStream):

    def write(self, item):
        item.offset = self.output.tell()
        data = bytes(item)
        self.buffer.write(data)
        if self.buffer.tell() == self.size:
            self.output.write(zlib.compress(self.buffer.getvalue()))
            self.buffer.seek(0)
            self.buffer.truncate(0)

    def flush(self):
        self.output.write(zlib.compress(self.buffer.getvalue()))
        self.buffer.seek(0)
        self.buffer.truncate(0)