from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
def _subgraph_parse(self, node, pathnode, extra_blocks):
    """parse the body and any `else` block of `if` and `for` statements"""
    loose_ends = []
    self.tail = pathnode
    self.dispatch_list(node.body)
    loose_ends.append(self.tail)
    for extra in extra_blocks:
        self.tail = pathnode
        self.dispatch_list(extra.body)
        loose_ends.append(self.tail)
    if node.orelse:
        self.tail = pathnode
        self.dispatch_list(node.orelse)
        loose_ends.append(self.tail)
    else:
        loose_ends.append(pathnode)
    if pathnode:
        bottom = PathNode('', look='point')
        for le in loose_ends:
            self.graph.connect(le, bottom)
        self.tail = bottom