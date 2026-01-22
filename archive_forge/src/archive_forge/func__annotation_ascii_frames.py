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
def _annotation_ascii_frames(sent):
    """
    ASCII string rendering of the sentence along with its targets and frame names.
    Called for all full-text sentences, as well as the few LU sentences with multiple
    targets (e.g., fn.lu(6412).exemplars[82] has two want.v targets).
    Line-wrapped to limit the display width.
    """
    overt = []
    for a, aset in enumerate(sent.annotationSet[1:]):
        for j, k in aset.Target:
            indexS = f'[{a + 1}]'
            if aset.status == 'UNANN' or aset.LU.status == 'Problem':
                indexS += ' '
                if aset.status == 'UNANN':
                    indexS += '!'
                if aset.LU.status == 'Problem':
                    indexS += '?'
            overt.append((j, k, aset.LU.frame.name, indexS))
    overt = sorted(overt)
    duplicates = set()
    for o, (j, k, fname, asetIndex) in enumerate(overt):
        if o > 0 and j <= overt[o - 1][1]:
            if overt[o - 1][:2] == (j, k) and overt[o - 1][2] == fname:
                combinedIndex = overt[o - 1][3] + asetIndex
                combinedIndex = combinedIndex.replace(' !', '! ').replace(' ?', '? ')
                overt[o - 1] = overt[o - 1][:3] + (combinedIndex,)
                duplicates.add(o)
            else:
                s = sent.text
                for j, k, fname, asetIndex in overt:
                    s += '\n' + asetIndex + ' ' + sent.text[j:k] + ' :: ' + fname
                s += '\n(Unable to display sentence with targets marked inline due to overlap)'
                return s
    for o in reversed(sorted(duplicates)):
        del overt[o]
    s0 = sent.text
    s1 = ''
    s11 = ''
    s2 = ''
    i = 0
    adjust = 0
    fAbbrevs = OrderedDict()
    for j, k, fname, asetIndex in overt:
        if not j >= i:
            assert j >= i, ('Overlapping targets?' + (' UNANN' if any((aset.status == 'UNANN' for aset in sent.annotationSet[1:])) else ''), (j, k, asetIndex))
        s1 += ' ' * (j - i) + '*' * (k - j)
        short = fname[:k - j]
        if k - j < len(fname):
            r = 0
            while short in fAbbrevs:
                if fAbbrevs[short] == fname:
                    break
                r += 1
                short = fname[:k - j - 1] + str(r)
            else:
                fAbbrevs[short] = fname
        s11 += ' ' * (j - i) + short.ljust(k - j)
        if len(asetIndex) > k - j:
            amt = len(asetIndex) - (k - j)
            s0 = s0[:k + adjust] + '~' * amt + s0[k + adjust:]
            s1 = s1[:k + adjust] + ' ' * amt + s1[k + adjust:]
            s11 = s11[:k + adjust] + ' ' * amt + s11[k + adjust:]
            adjust += amt
        s2 += ' ' * (j - i) + asetIndex.ljust(k - j)
        i = k
    long_lines = [s0, s1, s11, s2]
    outstr = '\n\n'.join(map('\n'.join, zip_longest(*mimic_wrap(long_lines), fillvalue=' '))).replace('~', ' ')
    outstr += '\n'
    if fAbbrevs:
        outstr += ' (' + ', '.join(('='.join(pair) for pair in fAbbrevs.items())) + ')'
        assert len(fAbbrevs) == len(dict(fAbbrevs)), 'Abbreviation clash'
    return outstr