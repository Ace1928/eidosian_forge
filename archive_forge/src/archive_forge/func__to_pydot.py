import sys
from copy import deepcopy
from typing import List, Callable, Iterator, Union, Optional, Generic, TypeVar, TYPE_CHECKING
from collections import OrderedDict
def _to_pydot(subtree):
    color = hash(subtree.data) & 16777215
    color |= 8421504
    subnodes = [_to_pydot(child) if isinstance(child, Tree) else new_leaf(child) for child in subtree.children]
    node = pydot.Node(i[0], style='filled', fillcolor='#%x' % color, label=subtree.data)
    i[0] += 1
    graph.add_node(node)
    for subnode in subnodes:
        graph.add_edge(pydot.Edge(node, subnode))
    return node