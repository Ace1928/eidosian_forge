import collections
import io
import itertools
import os
from taskflow.types import graph
from taskflow.utils import iter_utils
from taskflow.utils import misc
class _DFSIter(object):
    """Depth first iterator (non-recursive) over the child nodes."""

    def __init__(self, root, include_self=False, right_to_left=True):
        self.root = root
        self.right_to_left = bool(right_to_left)
        self.include_self = bool(include_self)

    def __iter__(self):
        stack = []
        if self.include_self:
            stack.append(self.root)
        elif self.right_to_left:
            stack.extend(self.root.reverse_iter())
        else:
            stack.extend(iter(self.root))
        while stack:
            node = stack.pop()
            yield node
            if self.right_to_left:
                stack.extend(node.reverse_iter())
            else:
                stack.extend(iter(node))