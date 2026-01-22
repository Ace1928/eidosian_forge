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
class MarkClosureVisitor(CythonTransform):

    def visit_ModuleNode(self, node):
        self.needs_closure = False
        self.excludes = []
        self.visitchildren(node)
        return node

    def visit_FuncDefNode(self, node):
        self.needs_closure = False
        self.visitchildren(node)
        node.needs_closure = self.needs_closure
        self.needs_closure = True
        collector = YieldNodeCollector(self.excludes)
        collector.visitchildren(node)
        if node.is_async_def:
            coroutine_type = Nodes.AsyncDefNode
            if collector.has_yield:
                coroutine_type = Nodes.AsyncGenNode
                for yield_expr in collector.yields + collector.returns:
                    yield_expr.in_async_gen = True
            elif self.current_directives['iterable_coroutine']:
                coroutine_type = Nodes.IterableAsyncDefNode
        elif collector.has_await:
            found = next((y for y in collector.yields if y.is_await))
            error(found.pos, "'await' not allowed in generators (use 'yield')")
            return node
        elif collector.has_yield:
            coroutine_type = Nodes.GeneratorDefNode
        else:
            return node
        for i, yield_expr in enumerate(collector.yields, 1):
            yield_expr.label_num = i
        for retnode in collector.returns + collector.finallys + collector.excepts:
            retnode.in_generator = True
        gbody = Nodes.GeneratorBodyDefNode(pos=node.pos, name=node.name, body=node.body, is_async_gen_body=node.is_async_def and collector.has_yield)
        coroutine = coroutine_type(pos=node.pos, name=node.name, args=node.args, star_arg=node.star_arg, starstar_arg=node.starstar_arg, doc=node.doc, decorators=node.decorators, gbody=gbody, lambda_name=node.lambda_name, return_type_annotation=node.return_type_annotation, is_generator_expression=node.is_generator_expression)
        return coroutine

    def visit_CFuncDefNode(self, node):
        self.needs_closure = False
        self.visitchildren(node)
        node.needs_closure = self.needs_closure
        self.needs_closure = True
        if node.needs_closure and node.overridable:
            error(node.pos, 'closures inside cpdef functions not yet supported')
        return node

    def visit_LambdaNode(self, node):
        self.needs_closure = False
        self.visitchildren(node)
        node.needs_closure = self.needs_closure
        self.needs_closure = True
        return node

    def visit_ClassDefNode(self, node):
        self.visitchildren(node)
        self.needs_closure = True
        return node

    def visit_GeneratorExpressionNode(self, node):
        excludes = self.excludes
        if isinstance(node.loop, Nodes._ForInStatNode):
            self.excludes = [node.loop.iterator]
        node = self.visit_LambdaNode(node)
        self.excludes = excludes
        if not isinstance(node.loop, Nodes._ForInStatNode):
            return node
        itseq = node.loop.iterator.sequence
        if itseq.is_literal:
            return node
        _GeneratorExpressionArgumentsMarker(node).visit(itseq)
        return node