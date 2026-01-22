important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class Keyword(_LeafWithoutNewlines, _StringComparisonMixin):
    type = 'keyword'
    __slots__ = ()