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
def _exemplar_of_fes(self, ex, fes=None):
    """
        Given an exemplar sentence and a set of FE names, return the subset of FE names
        that are realized overtly in the sentence on the FE, FE2, or FE3 layer.

        If 'fes' is None, returns all overt FE names.
        """
    overtNames = set(list(zip(*ex.FE[0]))[2]) if ex.FE[0] else set()
    if 'FE2' in ex:
        overtNames |= set(list(zip(*ex.FE2[0]))[2]) if ex.FE2[0] else set()
        if 'FE3' in ex:
            overtNames |= set(list(zip(*ex.FE3[0]))[2]) if ex.FE3[0] else set()
    return overtNames & fes if fes is not None else overtNames