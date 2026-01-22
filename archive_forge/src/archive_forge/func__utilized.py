import string
from xml.dom import Node
def _utilized(n, node, other_attrs, unsuppressedPrefixes):
    """_utilized(n, node, other_attrs, unsuppressedPrefixes) -> boolean
    Return true if that nodespace is utilized within the node"""
    if n.startswith('xmlns:'):
        n = n[6:]
    elif n.startswith('xmlns'):
        n = n[5:]
    if n == '' and node.prefix in ['#default', None] or n == node.prefix or n in unsuppressedPrefixes:
        return 1
    for attr in other_attrs:
        if n == attr.prefix:
            return 1
    if unsuppressedPrefixes is not None:
        for attr in _attrs(node):
            if n == attr.prefix:
                return 1
    return 0