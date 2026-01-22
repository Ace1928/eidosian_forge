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
def _pretty_frame_relation_type(freltyp):
    """
    Helper function for pretty-printing a frame relation type.

    :param freltyp: The frame relation type to be printed.
    :type freltyp: AttrDict
    :return: A nicely formatted string representation of the frame relation type.
    :rtype: str
    """
    outstr = '<frame relation type ({0.ID}): {0.superFrameName} -- {0.name} -> {0.subFrameName}>'.format(freltyp)
    return outstr