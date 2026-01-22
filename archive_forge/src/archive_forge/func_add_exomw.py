import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter
from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file
def add_exomw(self):
    """
        Add languages from Extended OMW

        >>> import nltk
        >>> from nltk.corpus import wordnet as wn
        >>> wn.add_exomw()
        >>> print(wn.synset('intrinsically.r.01').lemmas(lang="eng_wikt"))
        [Lemma('intrinsically.r.01.per_se'), Lemma('intrinsically.r.01.as_such')]
        """
    from nltk.corpus import extended_omw
    self.add_omw()
    self._exomw_reader = extended_omw
    self.add_provs(self._exomw_reader)