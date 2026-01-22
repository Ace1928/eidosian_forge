from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
class ImmutableNBestSentencePieceText(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        _sentencepiece.ImmutableNBestSentencePieceText_swiginit(self, _sentencepiece.new_ImmutableNBestSentencePieceText())
    __swig_destroy__ = _sentencepiece.delete_ImmutableNBestSentencePieceText

    def _nbests_size(self):
        return _sentencepiece.ImmutableNBestSentencePieceText__nbests_size(self)

    def _nbests(self, index):
        return _sentencepiece.ImmutableNBestSentencePieceText__nbests(self, index)

    def SerializeAsString(self):
        return _sentencepiece.ImmutableNBestSentencePieceText_SerializeAsString(self)

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

    @property
    def nbests(self):
        return ImmutableNBestSentencePieceText.ImmutableSentencePieceTextIterator(self)

    def __eq__(self, other):
        return self.SerializeAsString() == other.SerializeAsString()

    def __hash__(self):
        return hash(self.SerializeAsString())

    def __str__(self):
        return '\n'.join(['nbests {{\n{}}}'.format(str(x)) for x in self.nbests])
    __repr__ = __str__