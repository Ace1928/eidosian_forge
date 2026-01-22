from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
class ImmutableSentencePieceText(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        _sentencepiece.ImmutableSentencePieceText_swiginit(self, _sentencepiece.new_ImmutableSentencePieceText())
    __swig_destroy__ = _sentencepiece.delete_ImmutableSentencePieceText

    def _pieces_size(self):
        return _sentencepiece.ImmutableSentencePieceText__pieces_size(self)

    def _pieces(self, index):
        return _sentencepiece.ImmutableSentencePieceText__pieces(self, index)

    def _text(self):
        return _sentencepiece.ImmutableSentencePieceText__text(self)

    def _score(self):
        return _sentencepiece.ImmutableSentencePieceText__score(self)

    def SerializeAsString(self):
        return _sentencepiece.ImmutableSentencePieceText_SerializeAsString(self)

    def _text_as_bytes(self):
        return _sentencepiece.ImmutableSentencePieceText__text_as_bytes(self)
    text = property(_text)
    text_as_bytes = property(_text_as_bytes)
    score = property(_score)

    class ImmutableSentencePieceIterator:

        def __init__(self, proto):
            self.proto = proto
            self.len = self.proto._pieces_size()

        def __len__(self):
            return self.len

        def __getitem__(self, index):
            if isinstance(index, slice):
                return [self.proto._pieces(i) for i in range(self.len)][index.start:index.stop:index.step]
            if index < 0:
                index = index + self.len
            if index < 0 or index >= self.len:
                raise IndexError('piece index is out of range')
            return self.proto._pieces(index)

        def __str__(self):
            return '\n'.join(['pieces {{\n{}}}'.format(str(x)) for x in self])
        __repr__ = __str__

    @property
    def pieces(self):
        return ImmutableSentencePieceText.ImmutableSentencePieceIterator(self)

    def __eq__(self, other):
        return self.SerializeAsString() == other.SerializeAsString()

    def __hash__(self):
        return hash(self.SerializeAsString())

    def __str__(self):
        return 'text: "{}"\nscore: {}\n{}'.format(self.text, self.score, '\n'.join(['pieces {{\n{}}}'.format(str(x)) for x in self.pieces]))
    __repr__ = __str__