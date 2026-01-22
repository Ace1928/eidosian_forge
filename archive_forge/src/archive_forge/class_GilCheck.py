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
class GilCheck(VisitorTransform):
    """
    Call `node.gil_check(env)` on each node to make sure we hold the
    GIL when we need it.  Raise an error when on Python operations
    inside a `nogil` environment.

    Additionally, raise exceptions for closely nested with gil or with nogil
    statements. The latter would abort Python.
    """

    def __call__(self, root):
        self.env_stack = [root.scope]
        self.nogil = False
        self.nogil_declarator_only = False
        self.current_gilstat_node_knows_gil_state = False
        return super(GilCheck, self).__call__(root)

    def _visit_scoped_children(self, node, gil_state):
        was_nogil = self.nogil
        outer_attrs = node.outer_attrs
        if outer_attrs and len(self.env_stack) > 1:
            self.nogil = self.env_stack[-2].nogil
            self.visitchildren(node, outer_attrs)
        self.nogil = gil_state
        self.visitchildren(node, attrs=None, exclude=outer_attrs)
        self.nogil = was_nogil

    def visit_FuncDefNode(self, node):
        self.env_stack.append(node.local_scope)
        inner_nogil = node.local_scope.nogil
        nogil_declarator_only = self.nogil_declarator_only
        if inner_nogil:
            self.nogil_declarator_only = True
        if inner_nogil and node.nogil_check:
            node.nogil_check(node.local_scope)
        self._visit_scoped_children(node, inner_nogil)
        self.nogil_declarator_only = nogil_declarator_only
        self.env_stack.pop()
        return node

    def visit_GILStatNode(self, node):
        if node.condition is not None:
            error(node.condition.pos, 'Non-constant condition in a `with %s(<condition>)` statement' % node.state)
            return node
        if self.nogil and node.nogil_check:
            node.nogil_check()
        was_nogil = self.nogil
        is_nogil = node.state == 'nogil'
        if was_nogil == is_nogil and (not self.nogil_declarator_only):
            if not was_nogil:
                error(node.pos, 'Trying to acquire the GIL while it is already held.')
            else:
                error(node.pos, 'Trying to release the GIL while it was previously released.')
        if self.nogil_declarator_only:
            node.scope_gil_state_known = False
        if isinstance(node.finally_clause, Nodes.StatListNode):
            node.finally_clause, = node.finally_clause.stats
        nogil_declarator_only = self.nogil_declarator_only
        self.nogil_declarator_only = False
        current_gilstat_node_knows_gil_state = self.current_gilstat_node_knows_gil_state
        self.current_gilstat_node_knows_gil_state = node.scope_gil_state_known
        self._visit_scoped_children(node, is_nogil)
        self.nogil_declarator_only = nogil_declarator_only
        self.current_gilstat_node_knows_gil_state = current_gilstat_node_knows_gil_state
        return node

    def visit_ParallelRangeNode(self, node):
        if node.nogil or self.nogil_declarator_only:
            node_was_nogil, node.nogil = (node.nogil, False)
            node = Nodes.GILStatNode(node.pos, state='nogil', body=node)
            if not node_was_nogil and self.nogil_declarator_only:
                node.scope_gil_state_known = False
            return self.visit_GILStatNode(node)
        if not self.nogil:
            error(node.pos, 'prange() can only be used without the GIL')
            return None
        node.nogil_check(self.env_stack[-1])
        self.visitchildren(node)
        return node

    def visit_ParallelWithBlockNode(self, node):
        if not self.nogil:
            error(node.pos, 'The parallel section may only be used without the GIL')
            return None
        if self.nogil_declarator_only:
            node = Nodes.GILStatNode(node.pos, state='nogil', body=node)
            node.scope_gil_state_known = False
            return self.visit_GILStatNode(node)
        if node.nogil_check:
            node.nogil_check(self.env_stack[-1])
        self.visitchildren(node)
        return node

    def visit_TryFinallyStatNode(self, node):
        """
        Take care of try/finally statements in nogil code sections.
        """
        if not self.nogil or isinstance(node, Nodes.GILStatNode):
            return self.visit_Node(node)
        node.nogil_check = None
        node.is_try_finally_in_nogil = True
        self.visitchildren(node)
        return node

    def visit_GILExitNode(self, node):
        if not self.current_gilstat_node_knows_gil_state:
            node.scope_gil_state_known = False
        self.visitchildren(node)
        return node

    def visit_Node(self, node):
        if self.env_stack and self.nogil and node.nogil_check:
            node.nogil_check(self.env_stack[-1])
        if node.outer_attrs:
            self._visit_scoped_children(node, self.nogil)
        else:
            self.visitchildren(node)
        if self.nogil:
            node.in_nogil_context = True
        return node