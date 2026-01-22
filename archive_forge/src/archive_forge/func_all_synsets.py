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
def all_synsets(self, pos=None, lang='eng'):
    """Iterate over all synsets with a given part of speech tag.
        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        """
    if lang == 'eng':
        return self.all_eng_synsets(pos=pos)
    else:
        return self.all_omw_synsets(pos=pos, lang=lang)