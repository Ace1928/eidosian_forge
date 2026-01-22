import nltk.data
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
class CategorizedPlaintextCorpusReader(CategorizedCorpusReader, PlaintextCorpusReader):
    """
    A reader for plaintext corpora whose documents are divided into
    categories based on their file identifiers.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining arguments
        are passed to the ``PlaintextCorpusReader`` constructor.
        """
        CategorizedCorpusReader.__init__(self, kwargs)
        PlaintextCorpusReader.__init__(self, *args, **kwargs)