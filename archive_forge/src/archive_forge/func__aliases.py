important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def _aliases(self):
    """
        :return list of Name: Returns all the alias
        """
    return dict(((alias, path[-1]) for path, alias in self._dotted_as_names() if alias is not None))