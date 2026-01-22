import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def _register_with_indexes(self, edge):
    """
        A helper function for ``insert``, which registers the new
        edge with all existing indexes.
        """
    for restr_keys, index in self._indexes.items():
        vals = tuple((getattr(edge, key)() for key in restr_keys))
        index.setdefault(vals, []).append(edge)