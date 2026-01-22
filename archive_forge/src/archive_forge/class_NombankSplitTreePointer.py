from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
class NombankSplitTreePointer(NombankPointer):

    def __init__(self, pieces):
        self.pieces = pieces
        'A list of the pieces that make up this chain.  Elements are\n           all ``NombankTreePointer`` pointers.'

    def __str__(self):
        return ','.join(('%s' % p for p in self.pieces))

    def __repr__(self):
        return '<NombankSplitTreePointer: %s>' % self

    def select(self, tree):
        if tree is None:
            raise ValueError('Parse tree not available')
        return Tree('*SPLIT*', [p.select(tree) for p in self.pieces])