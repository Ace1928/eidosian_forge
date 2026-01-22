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
class _ExtraIndex:
    __slots__ = ('indexField', 'maxFieldSize', 'fileOffset', 'chunks', 'get_value')
    formatter = struct.Struct('=xxHQxxxxHxx')

    def __init__(self, name, declaration):
        self.maxFieldSize = 0
        self.fileOffset = None
        for index, field in enumerate(declaration):
            if field.name == name:
                break
        else:
            raise ValueError("extraIndex field %s not a standard bed field or found in 'as' file.", name) from None
        if field.as_type != 'string':
            raise ValueError('Sorry for now can only index string fields.')
        self.indexField = index
        if name == 'chrom':
            self.get_value = lambda alignment: alignment.target.id
        elif name == 'name':
            self.get_value = lambda alignment: alignment.query.id
        else:
            self.get_value = lambda alignment: alignment.annotations[name]

    def updateMaxFieldSize(self, alignment):
        value = self.get_value(alignment)
        size = len(value)
        if size > self.maxFieldSize:
            self.maxFieldSize = size

    def addKeysFromRow(self, alignment, recordIx):
        value = self.get_value(alignment)
        self.chunks[recordIx]['name'] = value.encode()

    def addOffsetSize(self, offset, size, startIx, endIx):
        self.chunks[startIx:endIx]['offset'] = offset
        self.chunks[startIx:endIx]['size'] = size

    def __bytes__(self):
        indexFieldCount = 1
        return self.formatter.pack(indexFieldCount, self.fileOffset, self.indexField)