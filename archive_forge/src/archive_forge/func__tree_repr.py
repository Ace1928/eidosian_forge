import abc
import collections
import collections.abc
import operator
import sys
import typing
def _tree_repr(self, tree):
    cls, origin, metadata = tree
    if not isinstance(origin, tuple):
        tp_repr = typing._type_repr(origin)
    else:
        tp_repr = origin[0]._tree_repr(origin)
    metadata_reprs = ', '.join((repr(arg) for arg in metadata))
    return f'{cls}[{tp_repr}, {metadata_reprs}]'