from pythran.analyses.aliases import Aliases
from pythran.analyses.argument_effects import ArgumentEffects
from pythran.analyses.global_effects import GlobalEffects
from pythran.analyses.pure_functions import PureFunctions
from pythran.passmanager import ModuleAnalysis
from pythran.intrinsic import Intrinsic
import gast as ast
class PureExpressions(ModuleAnalysis):
    """Yields the set of pure expressions"""

    def __init__(self):
        self.result = set()
        super(PureExpressions, self).__init__(ArgumentEffects, GlobalEffects, Aliases, PureFunctions)

    def visit_FunctionDef(self, node):
        if node in self.pure_functions:
            self.result.add(node)
        for stmt in node.body:
            self.visit(stmt)

    def generic_visit(self, node):
        is_pure = all([self.visit(x) for x in ast.iter_child_nodes(node)])
        if is_pure:
            self.result.add(node)
        return is_pure

    def visit_Call(self, node):
        is_pure = all([self.visit(arg) for arg in node.args])
        func_aliases = self.aliases[node.func]
        if func_aliases:
            for func_alias in func_aliases:
                if isinstance(func_alias, Intrinsic):
                    is_pure &= not func_alias.global_effects
                else:
                    is_pure &= func_alias in self.pure_functions
                if func_alias in self.argument_effects:
                    func_aes = self.argument_effects[func_alias]
                    for arg, ae in zip(node.args, func_aes):
                        if ae:
                            try:
                                ast.literal_eval(arg)
                            except ValueError:
                                is_pure = False
                else:
                    is_pure = False
        else:
            is_pure = False
        is_pure &= self.visit(node.func)
        if is_pure:
            self.result.add(node)
        return is_pure