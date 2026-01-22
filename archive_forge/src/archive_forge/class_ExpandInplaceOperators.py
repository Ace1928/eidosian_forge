from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
class ExpandInplaceOperators(EnvTransform):

    def visit_InPlaceAssignmentNode(self, node):
        lhs = node.lhs
        rhs = node.rhs
        if lhs.type.is_cpp_class:
            return node
        if isinstance(lhs, ExprNodes.BufferIndexNode):
            return node
        env = self.current_env()

        def side_effect_free_reference(node, setting=False):
            if node.is_name:
                return (node, [])
            elif node.type.is_pyobject and (not setting):
                node = LetRefNode(node)
                return (node, [node])
            elif node.is_subscript:
                base, temps = side_effect_free_reference(node.base)
                index = LetRefNode(node.index)
                return (ExprNodes.IndexNode(node.pos, base=base, index=index), temps + [index])
            elif node.is_attribute:
                obj, temps = side_effect_free_reference(node.obj)
                return (ExprNodes.AttributeNode(node.pos, obj=obj, attribute=node.attribute), temps)
            elif isinstance(node, ExprNodes.BufferIndexNode):
                raise ValueError("Don't allow things like attributes of buffer indexing operations")
            else:
                node = LetRefNode(node)
                return (node, [node])
        try:
            lhs, let_ref_nodes = side_effect_free_reference(lhs, setting=True)
        except ValueError:
            return node
        dup = lhs.__class__(**lhs.__dict__)
        binop = ExprNodes.binop_node(node.pos, operator=node.operator, operand1=dup, operand2=rhs, inplace=True)
        lhs = lhs.analyse_target_types(env)
        dup.analyse_types(env)
        binop.analyse_operation(env)
        node = Nodes.SingleAssignmentNode(node.pos, lhs=lhs, rhs=binop.coerce_to(lhs.type, env))
        let_ref_nodes.reverse()
        for t in let_ref_nodes:
            node = LetNode(t, node)
        return node

    def visit_ExprNode(self, node):
        return node