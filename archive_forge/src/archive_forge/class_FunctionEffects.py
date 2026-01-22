from pythran.analyses.aliases import Aliases
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
import pythran.intrinsic as intrinsic
import gast as ast
from functools import reduce
class FunctionEffects(object):

    def __init__(self, node):
        self.func = node
        self.dependencies = lambda ctx: 0
        if isinstance(node, ast.FunctionDef):
            self.read_effects = [-1] * len(node.args.args)
        elif isinstance(node, intrinsic.Intrinsic):
            self.read_effects = [1 if isinstance(x, intrinsic.ReadOnceEffect) else 2 for x in node.argument_effects]
        elif isinstance(node, ast.alias):
            self.read_effects = []
        else:
            raise NotImplementedError