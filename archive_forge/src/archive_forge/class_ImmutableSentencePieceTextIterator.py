from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
class ImmutableSentencePieceTextIterator:

    def __init__(self, proto):
        self.proto = proto
        self.len = self.proto._nbests_size()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.proto._nbests(i) for i in range(self.len)][index.start:index.stop:index.step]
        if index < 0:
            index = index + self.len
        if index < 0 or index >= self.len:
            raise IndexError('nbests index is out of range')
        return self.proto._nbests(index)

    def __str__(self):
        return '\n'.join(['nbests {{\n{}}}'.format(str(x)) for x in self])
    __repr__ = __str__