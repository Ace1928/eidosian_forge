from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
class MarkOverflowingArithmetic(CythonTransform):
    might_overflow = False

    def __call__(self, root):
        self.env_stack = []
        self.env = root.scope
        return super(MarkOverflowingArithmetic, self).__call__(root)

    def visit_safe_node(self, node):
        self.might_overflow, saved = (False, self.might_overflow)
        self.visitchildren(node)
        self.might_overflow = saved
        return node

    def visit_neutral_node(self, node):
        self.visitchildren(node)
        return node

    def visit_dangerous_node(self, node):
        self.might_overflow, saved = (True, self.might_overflow)
        self.visitchildren(node)
        self.might_overflow = saved
        return node

    def visit_FuncDefNode(self, node):
        self.env_stack.append(self.env)
        self.env = node.local_scope
        self.visit_safe_node(node)
        self.env = self.env_stack.pop()
        return node

    def visit_NameNode(self, node):
        if self.might_overflow:
            entry = node.entry or self.env.lookup(node.name)
            if entry:
                entry.might_overflow = True
        return node

    def visit_BinopNode(self, node):
        if node.operator in '&|^':
            return self.visit_neutral_node(node)
        else:
            return self.visit_dangerous_node(node)

    def visit_SimpleCallNode(self, node):
        if node.function.is_name and node.function.name == 'abs':
            return self.visit_dangerous_node(node)
        else:
            return self.visit_neutral_node(node)
    visit_UnopNode = visit_neutral_node
    visit_UnaryMinusNode = visit_dangerous_node
    visit_InPlaceAssignmentNode = visit_dangerous_node
    visit_Node = visit_safe_node

    def visit_assignment(self, lhs, rhs):
        if isinstance(rhs, ExprNodes.IntNode) and isinstance(lhs, ExprNodes.NameNode) and Utils.long_literal(rhs.value):
            entry = lhs.entry or self.env.lookup(lhs.name)
            if entry:
                entry.might_overflow = True

    def visit_SingleAssignmentNode(self, node):
        self.visit_assignment(node.lhs, node.rhs)
        self.visitchildren(node)
        return node

    def visit_CascadedAssignmentNode(self, node):
        for lhs in node.lhs_list:
            self.visit_assignment(lhs, node.rhs)
        self.visitchildren(node)
        return node