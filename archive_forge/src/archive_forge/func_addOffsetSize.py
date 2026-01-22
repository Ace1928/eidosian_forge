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
def addOffsetSize(self, offset, size, startIx, endIx):
    self.chunks[startIx:endIx]['offset'] = offset
    self.chunks[startIx:endIx]['size'] = size