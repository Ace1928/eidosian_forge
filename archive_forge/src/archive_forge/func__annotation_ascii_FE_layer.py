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
def _annotation_ascii_FE_layer(overt, ni, feAbbrevs):
    """Helper for _annotation_ascii_FEs()."""
    s1 = ''
    s2 = ''
    i = 0
    for j, k, fename in overt:
        s1 += ' ' * (j - i) + ('^' if fename.islower() else '-') * (k - j)
        short = fename[:k - j]
        if len(fename) > len(short):
            r = 0
            while short in feAbbrevs:
                if feAbbrevs[short] == fename:
                    break
                r += 1
                short = fename[:k - j - 1] + str(r)
            else:
                feAbbrevs[short] = fename
        s2 += ' ' * (j - i) + short.ljust(k - j)
        i = k
    sNI = ''
    if ni:
        sNI += ' [' + ', '.join((':'.join(x) for x in sorted(ni.items()))) + ']'
    return [s1, s2, sNI]