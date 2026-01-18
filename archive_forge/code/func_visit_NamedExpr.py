from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def visit_NamedExpr(self, node):
    dnode = self.chains.setdefault(node, Def(node))
    self.visit(node.value).add_user(dnode)
    self.visit(node.target)
    return dnode