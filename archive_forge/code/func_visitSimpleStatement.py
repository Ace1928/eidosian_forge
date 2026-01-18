from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
def visitSimpleStatement(self, node):
    if node.lineno is None:
        lineno = 0
    else:
        lineno = node.lineno
    name = 'Stmt %d' % lineno
    self.appendPathNode(name)