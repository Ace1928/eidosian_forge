from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
def visitIf(self, node):
    name = 'If %d' % node.lineno
    self._subgraph(node, name)