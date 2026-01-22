import functools
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, concat
def _append_space(self, sentence):
    if self.show_tags:
        sentence.append((' ', 'space'))
    else:
        sentence.append(' ')