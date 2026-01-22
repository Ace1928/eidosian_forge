from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
def _read_tuple_block(self, stream):
    line = stream.readline().strip()
    if line:
        return [tuple(line.split(self._delimiter, 1))]
    else:
        return []