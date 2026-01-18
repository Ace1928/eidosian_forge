from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def visit_excepthandler(self, node):
    dnode = self.chains.setdefault(node, Def(node))
    if node.type:
        self.visit(node.type).add_user(dnode)
    if node.name:
        self.visit(node.name).add_user(dnode)
    self.process_body(node.body)
    return dnode