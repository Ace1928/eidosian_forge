from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
def visitWith(self, node):
    name = 'With %d' % node.lineno
    self.appendPathNode(name)
    self.dispatch_list(node.body)