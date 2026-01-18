from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree

        Convert this pointer to a standard 'tree position' pointer,
        given that it points to the given tree.
        