from heapq import heappop, heappush
from itertools import count
import networkx as nx
def _merge_children(self, root):
    """Merge the subtrees of the root using the standard two-pass method.
        The resulting subtree is detached from the root.
        """
    node = root.left
    root.left = None
    if node is not None:
        link = self._link
        prev = None
        while True:
            next = node.next
            if next is None:
                node.prev = prev
                break
            next_next = next.next
            node = link(node, next)
            node.prev = prev
            prev = node
            if next_next is None:
                break
            node = next_next
        prev = node.prev
        while prev is not None:
            prev_prev = prev.prev
            node = link(prev, node)
            prev = prev_prev
        node.prev = None
        node.next = None
        node.parent = None
    return node