import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class LeafEdge(EdgeI):
    """
    An edge that records the fact that a leaf value is consistent with
    a word in the sentence.  A leaf edge consists of:

    - An index, indicating the position of the word.
    - A leaf, specifying the word's content.

    A leaf edge's left-hand side is its leaf value, and its right hand
    side is ``()``.  Its span is ``[index, index+1]``, and its dot
    position is ``0``.
    """

    def __init__(self, leaf, index):
        """
        Construct a new ``LeafEdge``.

        :param leaf: The new edge's leaf value, specifying the word
            that is recorded by this edge.
        :param index: The new edge's index, specifying the position of
            the word that is recorded by this edge.
        """
        self._leaf = leaf
        self._index = index
        self._comparison_key = (leaf, index)

    def lhs(self):
        return self._leaf

    def span(self):
        return (self._index, self._index + 1)

    def start(self):
        return self._index

    def end(self):
        return self._index + 1

    def length(self):
        return 1

    def rhs(self):
        return ()

    def dot(self):
        return 0

    def is_complete(self):
        return True

    def is_incomplete(self):
        return False

    def nextsym(self):
        return None

    def __str__(self):
        return f'[{self._index}:{self._index + 1}] {repr(self._leaf)}'

    def __repr__(self):
        return '[Edge: %s]' % self