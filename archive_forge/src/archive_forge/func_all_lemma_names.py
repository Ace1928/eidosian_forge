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
def all_lemma_names(self, pos=None, lang='eng'):
    """Return all lemma names for all synsets for the given
        part of speech tag and language or languages. If pos is
        not specified, all synsets for all parts of speech will
        be used."""
    if lang == 'eng':
        if pos is None:
            return iter(self._lemma_pos_offset_map)
        else:
            return (lemma for lemma in self._lemma_pos_offset_map if pos in self._lemma_pos_offset_map[lemma])
    else:
        self._load_lang_data(lang)
        lemma = []
        for i in self._lang_data[lang][0]:
            if pos is not None and i[-1] != pos:
                continue
            lemma.extend(self._lang_data[lang][0][i])
        lemma = iter(set(lemma))
        return lemma