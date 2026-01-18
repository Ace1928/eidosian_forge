from pythran.analyses import PureExpressions, DefUseChains, Ancestors
from pythran.openmp import OMPDirective
from pythran.passmanager import Transformation
import pythran.metadata as metadata
import gast as ast
def used_target(self, node):
    if isinstance(node, ast.Name):
        if node.id in self.blacklist:
            return True
        chain = self.def_use_chains.chains[node]
        return bool(chain.users())
    return True