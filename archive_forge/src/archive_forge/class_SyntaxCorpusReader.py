import os
import re
from collections import defaultdict
from itertools import chain
from nltk.corpus.reader.util import *
from nltk.data import FileSystemPathPointer, PathPointer, ZipFilePathPointer
class SyntaxCorpusReader(CorpusReader):
    """
    An abstract base class for reading corpora consisting of
    syntactically parsed text.  Subclasses should define:

      - ``__init__``, which specifies the location of the corpus
        and a method for detecting the sentence blocks in corpus files.
      - ``_read_block``, which reads a block from the input stream.
      - ``_word``, which takes a block and returns a list of list of words.
      - ``_tag``, which takes a block and returns a list of list of tagged
        words.
      - ``_parse``, which takes a block and returns a list of parsed
        sentences.
    """

    def _parse(self, s):
        raise NotImplementedError()

    def _word(self, s):
        raise NotImplementedError()

    def _tag(self, s):
        raise NotImplementedError()

    def _read_block(self, stream):
        raise NotImplementedError()

    def parsed_sents(self, fileids=None):
        reader = self._read_parsed_sent_block
        return concat([StreamBackedCorpusView(fileid, reader, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def tagged_sents(self, fileids=None, tagset=None):

        def reader(stream):
            return self._read_tagged_sent_block(stream, tagset)
        return concat([StreamBackedCorpusView(fileid, reader, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def sents(self, fileids=None):
        reader = self._read_sent_block
        return concat([StreamBackedCorpusView(fileid, reader, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def tagged_words(self, fileids=None, tagset=None):

        def reader(stream):
            return self._read_tagged_word_block(stream, tagset)
        return concat([StreamBackedCorpusView(fileid, reader, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def words(self, fileids=None):
        return concat([StreamBackedCorpusView(fileid, self._read_word_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def _read_word_block(self, stream):
        return list(chain.from_iterable(self._read_sent_block(stream)))

    def _read_tagged_word_block(self, stream, tagset=None):
        return list(chain.from_iterable(self._read_tagged_sent_block(stream, tagset)))

    def _read_sent_block(self, stream):
        return list(filter(None, [self._word(t) for t in self._read_block(stream)]))

    def _read_tagged_sent_block(self, stream, tagset=None):
        return list(filter(None, [self._tag(t, tagset) for t in self._read_block(stream)]))

    def _read_parsed_sent_block(self, stream):
        return list(filter(None, [self._parse(t) for t in self._read_block(stream)]))