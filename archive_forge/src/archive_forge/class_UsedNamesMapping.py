important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class UsedNamesMapping(Mapping):
    """
    This class exists for the sole purpose of creating an immutable dict.
    """

    def __init__(self, dct):
        self._dict = dct

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other