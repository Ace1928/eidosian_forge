import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def _buildcorpusindex(self):
    self._fulltext_idx = {}
    with XMLCorpusView(self.abspath('fulltextIndex.xml'), 'fulltextIndex/corpus', self._handle_fulltextindex_elt) as view:
        for doclist in view:
            for doc in doclist:
                self._fulltext_idx[doc.ID] = doc