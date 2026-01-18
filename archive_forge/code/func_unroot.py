import random
import sys
from . import Nodes
def unroot(self):
    """Define a unrooted Tree structure, using data of a rooted Tree."""

    def _get_branches(node):
        branches = []
        for b in self.node(node).succ:
            branches.append([node, b, self.node(b).data.branchlength, self.node(b).data.support])
            branches.extend(_get_branches(b))
        return branches
    self.unrooted = _get_branches(self.root)
    if len(self.node(self.root).succ) == 2:
        rootbranches = [b for b in self.unrooted if self.root in b[:2]]
        b1 = self.unrooted.pop(self.unrooted.index(rootbranches[0]))
        b2 = self.unrooted.pop(self.unrooted.index(rootbranches[1]))
        newbranch = [b1[1], b2[1], b1[2] + b2[2]]
        if b1[3] is None:
            newbranch.append(b2[3])
        elif b2[3] is None:
            newbranch.append(b1[3])
        elif b1[3] == b2[3]:
            newbranch.append(b1[3])
        elif b1[3] == 0 or b2[3] == 0:
            newbranch.append(b1[3] + b2[3])
        else:
            raise TreeError('Support mismatch in bifurcating root: %f, %f' % (float(b1[3]), float(b2[3])))
        self.unrooted.append(newbranch)