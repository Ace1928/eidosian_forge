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
def updateMaxFieldSize(self, alignment):
    value = self.get_value(alignment)
    size = len(value)
    if size > self.maxFieldSize:
        self.maxFieldSize = size