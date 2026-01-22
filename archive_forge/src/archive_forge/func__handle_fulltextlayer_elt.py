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
def _handle_fulltextlayer_elt(self, elt):
    """Load information from the given 'layer' element. Each
        'layer' contains several "label" elements."""
    info = self._load_xml_attributes(AttrDict(), elt)
    info['_type'] = 'layer'
    info['label'] = []
    for sub in elt:
        if sub.tag.endswith('label'):
            l = self._load_xml_attributes(AttrDict(), sub)
            info['label'].append(l)
    return info