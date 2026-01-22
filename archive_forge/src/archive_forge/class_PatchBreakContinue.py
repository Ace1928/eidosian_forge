from pythran.analyses import (ImportedIds, HasReturn, IsAssigned, CFG,
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
class PatchBreakContinue(ast.NodeTransformer):

    def __init__(self, guard):
        self.guard = guard

    def visit_For(self, _):
        pass

    def visit_While(self, _):
        pass

    def patch_Control(self, node, flag):
        new_node = deepcopy(self.guard)
        ret_val = new_node.value
        if isinstance(ret_val, ast.Call):
            if flag == LOOP_BREAK:
                ret_val.func.attr = 'StaticIfBreak'
            else:
                ret_val.func.attr = 'StaticIfCont'
        else:
            new_node.value.elts[0].value = flag
        return new_node

    def visit_Break(self, node):
        return self.patch_Control(node, LOOP_BREAK)

    def visit_Continue(self, node):
        return self.patch_Control(node, LOOP_CONT)