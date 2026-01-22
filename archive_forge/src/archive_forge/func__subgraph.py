from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
def _subgraph(self, node, name, extra_blocks=()):
    """create the subgraphs representing any `if` and `for` statements"""
    if self.graph is None:
        self.graph = PathGraph(name, name, node.lineno, node.col_offset)
        pathnode = PathNode(name)
        self._subgraph_parse(node, pathnode, extra_blocks)
        self.graphs['%s%s' % (self.classname, name)] = self.graph
        self.reset()
    else:
        pathnode = self.appendPathNode(name)
        self._subgraph_parse(node, pathnode, extra_blocks)