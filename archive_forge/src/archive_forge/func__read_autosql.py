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
def _read_autosql(self, stream, header):
    autoSqlSize = header.totalSummaryOffset - header.autoSqlOffset
    fieldCount = header.fieldCount
    self.bedN = header.definedFieldCount
    stream.seek(header.autoSqlOffset)
    data = stream.read(autoSqlSize)
    declaration = AutoSQLTable.from_bytes(data)
    self._analyze_fields(declaration, fieldCount, self.bedN)
    return declaration