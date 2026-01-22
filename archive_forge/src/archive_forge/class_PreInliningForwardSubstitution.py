from pythran.analyses import LazynessAnalysis, UseDefChains, DefUseChains
from pythran.analyses import Literals, Ancestors, Identifiers, CFG, IsAssigned
from pythran.passmanager import Transformation
import pythran.graph as graph
from collections import defaultdict
import gast as ast
class PreInliningForwardSubstitution(ForwardSubstitution):
    """
    Replace variable that can be computed later, but only if this leads to a
    one-liner that's going to be a great inlining candidate.
    """

    def visit_FunctionDef(self, node):
        if all((isinstance(s, (ast.Return, ast.Assign)) for s in node.body)):
            r = super(PreInliningForwardSubstitution, self).visit_FunctionDef(node)
            return r
        return node