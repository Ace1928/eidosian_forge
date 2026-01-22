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
def _data_file(self, pos):
    """
        Return an open file pointer for the data file for the given
        part of speech.
        """
    if pos == ADJ_SAT:
        pos = ADJ
    if self._data_file_map.get(pos) is None:
        fileid = 'data.%s' % self._FILEMAP[pos]
        self._data_file_map[pos] = self.open(fileid)
    return self._data_file_map[pos]