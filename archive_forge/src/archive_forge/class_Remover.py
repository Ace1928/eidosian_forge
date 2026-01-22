from pythran.analyses import LazynessAnalysis, UseDefChains, DefUseChains
from pythran.analyses import Literals, Ancestors, Identifiers, CFG, IsAssigned
from pythran.passmanager import Transformation
import pythran.graph as graph
from collections import defaultdict
import gast as ast
class Remover(ast.NodeTransformer):

    def __init__(self, nodes):
        self.nodes = nodes

    def visit_Assign(self, node):
        if node in self.nodes:
            to_prune = self.nodes[node]
            node.targets = [tgt for tgt in node.targets if tgt not in to_prune]
            if node.targets:
                return node
            else:
                return ast.Pass()
        return node