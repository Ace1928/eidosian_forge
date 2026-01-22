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
def _pretty_fe(fe):
    """
    Helper function for pretty-printing a frame element.

    :param fe: The frame element to be printed.
    :type fe: AttrDict
    :return: A nicely formatted string representation of the frame element.
    :rtype: str
    """
    fekeys = fe.keys()
    outstr = ''
    outstr += 'frame element ({0.ID}): {0.name}\n    of {1.name}({1.ID})\n'.format(fe, fe.frame)
    if 'definition' in fekeys:
        outstr += '[definition]\n'
        outstr += _pretty_longstring(fe.definition, '  ')
    if 'abbrev' in fekeys:
        outstr += f'[abbrev] {fe.abbrev}\n'
    if 'coreType' in fekeys:
        outstr += f'[coreType] {fe.coreType}\n'
    if 'requiresFE' in fekeys:
        outstr += '[requiresFE] '
        if fe.requiresFE is None:
            outstr += '<None>\n'
        else:
            outstr += f'{fe.requiresFE.name}({fe.requiresFE.ID})\n'
    if 'excludesFE' in fekeys:
        outstr += '[excludesFE] '
        if fe.excludesFE is None:
            outstr += '<None>\n'
        else:
            outstr += f'{fe.excludesFE.name}({fe.excludesFE.ID})\n'
    if 'semType' in fekeys:
        outstr += '[semType] '
        if fe.semType is None:
            outstr += '<None>\n'
        else:
            outstr += '\n  ' + f'{fe.semType.name}({fe.semType.ID})' + '\n'
    return outstr