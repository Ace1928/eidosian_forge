from collections import OrderedDict
from .Node import Node
@staticmethod
def treeCopy(tree):
    copy = OrderedDict()
    for k, v in tree.items():
        if isNodeClass(v):
            copy[k] = v
        else:
            copy[k] = NodeLibrary.treeCopy(v)
    return copy