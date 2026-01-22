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
def _collapse_ambig(self, children):
    new_children = []
    for child in children:
        if hasattr(child, 'data') and child.data == '_ambig':
            new_children += child.children
        else:
            new_children.append(child)
    return new_children