import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _visit_internal(self, mode):
    """Visits the CFG, breadth-first."""
    assert mode in (_WalkMode.FORWARD, _WalkMode.REVERSE)
    if mode == _WalkMode.FORWARD:
        open_ = [self.graph.entry]
    elif mode == _WalkMode.REVERSE:
        open_ = list(self.graph.exit)
    closed = set()
    while open_:
        node = open_.pop(0)
        closed.add(node)
        should_revisit = self.visit_node(node)
        if mode == _WalkMode.FORWARD:
            children = node.next
        elif mode == _WalkMode.REVERSE:
            children = node.prev
        for next_ in children:
            if should_revisit or next_ not in closed:
                open_.append(next_)