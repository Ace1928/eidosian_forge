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
def _pretty_fe_relation(ferel):
    """
    Helper function for pretty-printing an FE relation.

    :param ferel: The FE relation to be printed.
    :type ferel: AttrDict
    :return: A nicely formatted string representation of the FE relation.
    :rtype: str
    """
    outstr = '<{0.type.superFrameName}={0.frameRelation.superFrameName}.{0.superFEName} -- {0.type.name} -> {0.type.subFrameName}={0.frameRelation.subFrameName}.{0.subFEName}>'.format(ferel)
    return outstr