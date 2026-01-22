important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class ForStmt(Flow):
    type = 'for_stmt'
    __slots__ = ()

    def get_testlist(self):
        """
        Returns the input node ``y`` from: ``for x in y:``.
        """
        return self.children[3]

    def get_defined_names(self, include_setitem=False):
        return _defined_names(self.children[1], include_setitem)