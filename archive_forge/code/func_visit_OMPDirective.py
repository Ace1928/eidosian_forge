from pythran.analyses import PureExpressions, DefUseChains, Ancestors
from pythran.openmp import OMPDirective
from pythran.passmanager import Transformation
import pythran.metadata as metadata
import gast as ast
def visit_OMPDirective(self, node):
    for dep in node.deps:
        if isinstance(dep, ast.Name):
            self.blacklist.add(dep.id)
    return node