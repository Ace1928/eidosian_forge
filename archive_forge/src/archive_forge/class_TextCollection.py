import re
import sys
from collections import Counter, defaultdict, namedtuple
from functools import reduce
from math import log
from nltk.collocations import BigramCollocationFinder
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.metrics import BigramAssocMeasures, f_measure
from nltk.probability import ConditionalFreqDist as CFD
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.util import LazyConcatenation, tokenwrap
class TextCollection(Text):
    """A collection of texts, which can be loaded with list of texts, or
    with a corpus consisting of one or more texts, and which supports
    counting, concordancing, collocation discovery, etc.  Initialize a
    TextCollection as follows:

    >>> import nltk.corpus
    >>> from nltk.text import TextCollection
    >>> from nltk.book import text1, text2, text3
    >>> gutenberg = TextCollection(nltk.corpus.gutenberg)
    >>> mytexts = TextCollection([text1, text2, text3])

    Iterating over a TextCollection produces all the tokens of all the
    texts in order.
    """

    def __init__(self, source):
        if hasattr(source, 'words'):
            source = [source.words(f) for f in source.fileids()]
        self._texts = source
        Text.__init__(self, LazyConcatenation(source))
        self._idf_cache = {}

    def tf(self, term, text):
        """The frequency of the term in text."""
        return text.count(term) / len(text)

    def idf(self, term):
        """The number of texts in the corpus divided by the
        number of texts that the term appears in.
        If a term does not appear in the corpus, 0.0 is returned."""
        idf = self._idf_cache.get(term)
        if idf is None:
            matches = len([True for text in self._texts if term in text])
            if len(self._texts) == 0:
                raise ValueError('IDF undefined for empty document collection')
            idf = log(len(self._texts) / matches) if matches else 0.0
            self._idf_cache[term] = idf
        return idf

    def tf_idf(self, term, text):
        return self.tf(term, text) * self.idf(term)