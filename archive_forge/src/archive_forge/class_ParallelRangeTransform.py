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
class ParallelRangeTransform(CythonTransform, SkipDeclarations):
    """
    Transform cython.parallel stuff. The parallel_directives come from the
    module node, set there by InterpretCompilerDirectives.

        x = cython.parallel.threadavailable()   -> ParallelThreadAvailableNode
        with nogil, cython.parallel.parallel(): -> ParallelWithBlockNode
            print cython.parallel.threadid()    -> ParallelThreadIdNode
            for i in cython.parallel.prange(...):  -> ParallelRangeNode
                ...
    """
    parallel_directive = None
    namenode_is_cython_module = False
    in_context_manager_section = False
    state = None
    directive_to_node = {u'cython.parallel.parallel': Nodes.ParallelWithBlockNode, u'cython.parallel.threadid': ExprNodes.ParallelThreadIdNode, u'cython.parallel.prange': Nodes.ParallelRangeNode}

    def node_is_parallel_directive(self, node):
        return node.name in self.parallel_directives or node.is_cython_module

    def get_directive_class_node(self, node):
        """
        Figure out which parallel directive was used and return the associated
        Node class.

        E.g. for a cython.parallel.prange() call we return ParallelRangeNode
        """
        if self.namenode_is_cython_module:
            directive = '.'.join(self.parallel_directive)
        else:
            directive = self.parallel_directives[self.parallel_directive[0]]
            directive = '%s.%s' % (directive, '.'.join(self.parallel_directive[1:]))
            directive = directive.rstrip('.')
        cls = self.directive_to_node.get(directive)
        if cls is None and (not (self.namenode_is_cython_module and self.parallel_directive[0] != 'parallel')):
            error(node.pos, 'Invalid directive: %s' % directive)
        self.namenode_is_cython_module = False
        self.parallel_directive = None
        return cls

    def visit_ModuleNode(self, node):
        """
        If any parallel directives were imported, copy them over and visit
        the AST
        """
        if node.parallel_directives:
            self.parallel_directives = node.parallel_directives
            return self.visit_Node(node)
        return node

    def visit_NameNode(self, node):
        if self.node_is_parallel_directive(node):
            self.parallel_directive = [node.name]
            self.namenode_is_cython_module = node.is_cython_module
        return node

    def visit_AttributeNode(self, node):
        self.visitchildren(node)
        if self.parallel_directive:
            self.parallel_directive.append(node.attribute)
        return node

    def visit_CallNode(self, node):
        self.visitchild(node, 'function')
        if not self.parallel_directive:
            self.visitchildren(node, exclude=('function',))
            return node
        if isinstance(node, ExprNodes.GeneralCallNode):
            args = node.positional_args.args
            kwargs = node.keyword_args
        else:
            args = node.args
            kwargs = {}
        parallel_directive_class = self.get_directive_class_node(node)
        if parallel_directive_class:
            node = parallel_directive_class(node.pos, args=args, kwargs=kwargs)
        return node

    def visit_WithStatNode(self, node):
        """Rewrite with cython.parallel.parallel() blocks"""
        newnode = self.visit(node.manager)
        if isinstance(newnode, Nodes.ParallelWithBlockNode):
            if self.state == 'parallel with':
                error(node.manager.pos, 'Nested parallel with blocks are disallowed')
            self.state = 'parallel with'
            body = self.visitchild(node, 'body')
            self.state = None
            newnode.body = body
            return newnode
        elif self.parallel_directive:
            parallel_directive_class = self.get_directive_class_node(node)
            if not parallel_directive_class:
                return None
            if parallel_directive_class is Nodes.ParallelWithBlockNode:
                error(node.pos, 'The parallel directive must be called')
                return None
        self.visitchild(node, 'body')
        return node

    def visit_ForInStatNode(self, node):
        """Rewrite 'for i in cython.parallel.prange(...):'"""
        self.visitchild(node, 'iterator')
        self.visitchild(node, 'target')
        in_prange = isinstance(node.iterator.sequence, Nodes.ParallelRangeNode)
        previous_state = self.state
        if in_prange:
            parallel_range_node = node.iterator.sequence
            parallel_range_node.target = node.target
            parallel_range_node.body = node.body
            parallel_range_node.else_clause = node.else_clause
            node = parallel_range_node
            if not isinstance(node.target, ExprNodes.NameNode):
                error(node.target.pos, 'Can only iterate over an iteration variable')
            self.state = 'prange'
        self.visitchild(node, 'body')
        self.state = previous_state
        self.visitchild(node, 'else_clause')
        return node

    def visit(self, node):
        """Visit a node that may be None"""
        if node is not None:
            return super(ParallelRangeTransform, self).visit(node)