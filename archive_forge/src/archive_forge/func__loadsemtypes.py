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
def _loadsemtypes(self):
    """Create the semantic types index."""
    self._semtypes = AttrDict()
    with XMLCorpusView(self.abspath('semTypes.xml'), 'semTypes/semType', self._handle_semtype_elt) as view:
        for st in view:
            n = st['name']
            a = st['abbrev']
            i = st['ID']
            self._semtypes[n] = i
            self._semtypes[a] = i
            self._semtypes[i] = st
    roots = []
    for st in self.semtypes():
        if st.superType:
            st.superType = self.semtype(st.superType.supID)
            st.superType.subTypes.append(st)
        else:
            if st not in roots:
                roots.append(st)
            st.rootType = st
    queue = list(roots)
    assert queue
    while queue:
        st = queue.pop(0)
        for child in st.subTypes:
            child.rootType = st.rootType
            queue.append(child)