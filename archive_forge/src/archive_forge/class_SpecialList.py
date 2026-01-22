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
class SpecialList(list):
    """
    A list subclass which adds a '_type' attribute for special printing
    (similar to an AttrDict, though this is NOT an AttrDict subclass).
    """

    def __init__(self, typ, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._type = typ

    def _str(self):
        outstr = ''
        assert self._type
        if len(self) == 0:
            outstr = '[]'
        elif self._type == 'luexemplars':
            outstr = _pretty_exemplars(self, self[0].LU)
        else:
            assert False, self._type
        return outstr

    def __str__(self):
        return self._str()

    def __repr__(self):
        return self.__str__()