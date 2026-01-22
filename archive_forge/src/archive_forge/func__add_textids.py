from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def _add_textids(self, file_id, text_id):
    self._f2t[file_id].append(text_id)
    self._t2f[text_id].append(file_id)