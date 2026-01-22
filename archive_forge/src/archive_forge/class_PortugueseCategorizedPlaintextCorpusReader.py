import nltk.data
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
class PortugueseCategorizedPlaintextCorpusReader(CategorizedPlaintextCorpusReader):

    def __init__(self, *args, **kwargs):
        CategorizedCorpusReader.__init__(self, kwargs)
        kwargs['sent_tokenizer'] = nltk.data.LazyLoader('tokenizers/punkt/portuguese.pickle')
        PlaintextCorpusReader.__init__(self, *args, **kwargs)