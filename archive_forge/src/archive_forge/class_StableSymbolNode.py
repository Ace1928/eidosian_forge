from typing import Type, AbstractSet
from random import randint
from collections import deque
from operator import attrgetter
from importlib import import_module
from functools import partial
from ..parse_tree_builder import AmbiguousIntermediateExpander
from ..visitors import Discard
from ..utils import logger, OrderedSet
from ..tree import Tree
class StableSymbolNode(SymbolNode):
    """A version of SymbolNode that uses OrderedSet for output stability"""
    Set = OrderedSet