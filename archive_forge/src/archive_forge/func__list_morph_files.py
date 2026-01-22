import functools
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, concat
def _list_morph_files(self, fileids):
    return [f for f in self.abspaths(fileids)]