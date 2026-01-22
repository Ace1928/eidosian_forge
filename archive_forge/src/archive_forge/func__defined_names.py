important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def _defined_names(current, include_setitem):
    """
    A helper function to find the defined names in statements, for loops and
    list comprehensions.
    """
    names = []
    if current.type in ('testlist_star_expr', 'testlist_comp', 'exprlist', 'testlist'):
        for child in current.children[::2]:
            names += _defined_names(child, include_setitem)
    elif current.type in ('atom', 'star_expr'):
        names += _defined_names(current.children[1], include_setitem)
    elif current.type in ('power', 'atom_expr'):
        if current.children[-2] != '**':
            trailer = current.children[-1]
            if trailer.children[0] == '.':
                names.append(trailer.children[1])
            elif trailer.children[0] == '[' and include_setitem:
                for node in current.children[-2::-1]:
                    if node.type == 'trailer':
                        names.append(node.children[1])
                        break
                    if node.type == 'name':
                        names.append(node)
                        break
    else:
        names.append(current)
    return names