from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def visit_Destructured(self, node):
    dnode = self.chains.setdefault(node, Def(node))
    tmp_store = ast.Store()
    for elt in node.elts:
        if isinstance(elt, ast.Name):
            tmp_store, elt.ctx = (elt.ctx, tmp_store)
            self.visit(elt)
            tmp_store, elt.ctx = (elt.ctx, tmp_store)
        elif isinstance(elt, ast.Subscript):
            self.visit(elt)
        elif isinstance(elt, (ast.List, ast.Tuple)):
            self.visit_Destructured(elt)
    return dnode