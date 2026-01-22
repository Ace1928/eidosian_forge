import random
import sys
from . import Nodes
def _connect_subtree(parent, child):
    """Attach subtree starting with node child to parent (PRIVATE)."""
    for i, branch in enumerate(self.unrooted):
        if parent in branch[:2] and child in branch[:2]:
            branch = self.unrooted.pop(i)
            break
    else:
        raise TreeError('Unable to connect nodes for rooting: nodes %d and %d are not connected' % (parent, child))
    self.link(parent, child)
    self.node(child).data.branchlength = branch[2]
    self.node(child).data.support = branch[3]
    child_branches = [b for b in self.unrooted if child in b[:2]]
    for b in child_branches:
        if child == b[0]:
            succ = b[1]
        else:
            succ = b[0]
        _connect_subtree(child, succ)