from . import (
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform
class _FindCFuncDefNode(TreeVisitor):
    """
    Finds the CFuncDefNode in the tree

    The assumption is that there's only one CFuncDefNode
    """
    found_node = None

    def visit_Node(self, node):
        if self.found_node:
            return
        else:
            self.visitchildren(node)

    def visit_CFuncDefNode(self, node):
        self.found_node = node

    def __call__(self, tree):
        self.visit(tree)
        return self.found_node