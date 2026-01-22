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
def addToCoverageDepth(self, alignment):
    start = alignment.coordinates[0, 0]
    end = alignment.coordinates[0, -1]
    if start > end:
        start, end = (end, start)
    existing = self.find(start, end)
    if existing is None:
        r = _Range(start, end, val=1)
        self.add(r)
    elif existing.start <= start and existing.end >= end:
        if existing.start < start:
            r = _Range(existing.start, start, existing.val)
            existing.start = start
            self.add(r)
        if existing.end > end:
            r = _Range(end, existing.end, existing.val)
            existing.end = end
            self.add(r)
        existing.val += 1
    else:
        items = list(self.root.traverse_range(start, end))
        s = start
        e = end
        for item in items:
            if s < item.start:
                r = _Range(s, item.start, 1)
                s = item.start
                self.add(r)
            elif s > item.start:
                r = _Range(item.start, s, item.val)
                item.start = s
                self.add(r)
            if item.start < end and item.end > end:
                r = _Range(end, item.end, item.val)
                item.end = end
                self.add(r)
            item.val += 1
            s = item.end
        if s < e:
            r = _Range(s, e, 1)
            self.add(r)