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
def _pretty_lu(lu):
    """
    Helper function for pretty-printing a lexical unit.

    :param lu: The lu to be printed.
    :type lu: AttrDict
    :return: A nicely formatted string representation of the lexical unit.
    :rtype: str
    """
    lukeys = lu.keys()
    outstr = ''
    outstr += 'lexical unit ({0.ID}): {0.name}\n\n'.format(lu)
    if 'definition' in lukeys:
        outstr += '[definition]\n'
        outstr += _pretty_longstring(lu.definition, '  ')
    if 'frame' in lukeys:
        outstr += f'\n[frame] {lu.frame.name}({lu.frame.ID})\n'
    if 'incorporatedFE' in lukeys:
        outstr += f'\n[incorporatedFE] {lu.incorporatedFE}\n'
    if 'POS' in lukeys:
        outstr += f'\n[POS] {lu.POS}\n'
    if 'status' in lukeys:
        outstr += f'\n[status] {lu.status}\n'
    if 'totalAnnotated' in lukeys:
        outstr += f'\n[totalAnnotated] {lu.totalAnnotated} annotated examples\n'
    if 'lexemes' in lukeys:
        outstr += '\n[lexemes] {}\n'.format(' '.join((f'{lex.name}/{lex.POS}' for lex in lu.lexemes)))
    if 'semTypes' in lukeys:
        outstr += f'\n[semTypes] {len(lu.semTypes)} semantic types\n'
        outstr += '  ' * (len(lu.semTypes) > 0) + ', '.join((f'{x.name}({x.ID})' for x in lu.semTypes)) + '\n' * (len(lu.semTypes) > 0)
    if 'URL' in lukeys:
        outstr += f'\n[URL] {lu.URL}\n'
    if 'subCorpus' in lukeys:
        subc = [x.name for x in lu.subCorpus]
        outstr += f'\n[subCorpus] {len(lu.subCorpus)} subcorpora\n'
        for line in textwrap.fill(', '.join(sorted(subc)), 60).split('\n'):
            outstr += f'  {line}\n'
    if 'exemplars' in lukeys:
        outstr += '\n[exemplars] {} sentences across all subcorpora\n'.format(len(lu.exemplars))
    return outstr