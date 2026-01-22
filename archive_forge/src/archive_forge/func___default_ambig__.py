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
def __default_ambig__(self, name, data):
    """Default operation on ambiguous rule (for override).

        Wraps data in an '_ambig_' node if it contains more than
        one element.
        """
    if len(data) > 1:
        return self.tree_class('_ambig', data)
    elif data:
        return data[0]
    return Discard