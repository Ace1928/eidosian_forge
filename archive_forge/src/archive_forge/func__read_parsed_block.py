import nltk
from nltk.corpus.reader.api import *
def _read_parsed_block(self, stream):
    return [self._parse(doc) for doc in self._read_block(stream) if self._parse(doc).docno is not None]