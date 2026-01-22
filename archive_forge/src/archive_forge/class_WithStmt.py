important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class WithStmt(Flow):
    type = 'with_stmt'
    __slots__ = ()

    def get_defined_names(self, include_setitem=False):
        """
        Returns the a list of `Name` that the with statement defines. The
        defined names are set after `as`.
        """
        names = []
        for with_item in self.children[1:-2:2]:
            if with_item.type == 'with_item':
                names += _defined_names(with_item.children[2], include_setitem)
        return names

    def get_test_node_from_name(self, name):
        node = name.search_ancestor('with_item')
        if node is None:
            raise ValueError('The name is not actually part of a with statement.')
        return node.children[0]