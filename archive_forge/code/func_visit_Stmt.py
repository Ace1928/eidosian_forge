from pythran.analyses import Inlinable, Aliases
from pythran.passmanager import Transformation
import gast as ast
import copy
def visit_Stmt(self, node):
    """ Add new variable definition before the Statement. """
    save_defs, self.defs = (self.defs or list(), list())
    self.generic_visit(node)
    new_defs, self.defs = (self.defs, save_defs)
    return new_defs + [node]