from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
def visitTryExcept(self, node):
    name = 'TryExcept %d' % node.lineno
    self._subgraph(node, name, extra_blocks=node.handlers)