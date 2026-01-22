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
def add_provs(self, reader):
    """Add languages from Multilingual Wordnet to the provenance dictionary"""
    fileids = reader.fileids()
    for fileid in fileids:
        prov, langfile = os.path.split(fileid)
        file_name, file_extension = os.path.splitext(langfile)
        if file_extension == '.tab':
            lang = file_name.split('-')[-1]
            if lang in self.provenances or prov in ['cldr', 'wikt']:
                lang = f'{lang}_{prov}'
            self.provenances[lang] = prov