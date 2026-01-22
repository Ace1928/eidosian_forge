import collections
import copy
import itertools
import random
import re
import warnings
def _filter_search(self, filter_func, order, follow_attrs):
    """Perform a BFS or DFS traversal through all elements in this tree (PRIVATE).

        :returns: generator of all elements for which ``filter_func`` is True.

        """
    order_opts = {'preorder': _preorder_traverse, 'postorder': _postorder_traverse, 'level': _level_traverse}
    try:
        order_func = order_opts[order]
    except KeyError:
        raise ValueError(f"Invalid order '{order}'; must be one of: {tuple(order_opts)}") from None
    if follow_attrs:
        get_children = _sorted_attrs
        root = self
    else:
        get_children = lambda elem: elem.clades
        root = self.root
    return filter(filter_func, order_func(root, get_children))