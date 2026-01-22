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
class _HandleGeneratorArguments(VisitorTransform, SkipDeclarations):

    def __call__(self, node):
        from . import Visitor
        assert isinstance(node, ExprNodes.GeneratorExpressionNode)
        self.gen_node = node
        self.args = list(node.def_node.args)
        self.call_parameters = list(node.call_parameters)
        self.tag_count = 0
        self.substitutions = {}
        self.visitchildren(node)
        for k, v in self.substitutions.items():
            Visitor.recursively_replace_node(node, k, v)
        node.def_node.args = self.args
        node.call_parameters = self.call_parameters
        return node

    def visit_GeneratorExpressionNode(self, node):
        new_node = self._handle_ExprNode(node, do_visit_children=False)
        return node

    def _handle_ExprNode(self, node, do_visit_children):
        if node.generator_arg_tag is not None and self.gen_node is not None and (self.gen_node == node.generator_arg_tag):
            pos = node.pos
            name_source = self.tag_count
            self.tag_count += 1
            name = EncodedString('.{0}'.format(name_source))
            def_node = self.gen_node.def_node
            if not def_node.local_scope.lookup_here(name):
                from . import Symtab
                cname = EncodedString(Naming.genexpr_arg_prefix + Symtab.punycodify_name(str(name_source)))
                name_decl = Nodes.CNameDeclaratorNode(pos=pos, name=name)
                type = node.type
                type = PyrexTypes.remove_cv_ref(type, remove_fakeref=False)
                name_decl.type = type
                new_arg = Nodes.CArgDeclNode(pos=pos, declarator=name_decl, base_type=None, default=None, annotation=None)
                new_arg.name = name_decl.name
                new_arg.type = type
                self.args.append(new_arg)
                node.generator_arg_tag = None
                self.call_parameters.append(node)
                new_arg.entry = def_node.declare_argument(def_node.local_scope, new_arg)
                new_arg.entry.cname = cname
                new_arg.entry.in_closure = True
            if do_visit_children:
                gen_node, self.gen_node = (self.gen_node, None)
                self.visitchildren(node)
                self.gen_node = gen_node
            name_node = ExprNodes.NameNode(pos, name=name, initialized_check=False)
            name_node.entry = self.gen_node.def_node.gbody.local_scope.lookup(name_node.name)
            name_node.type = name_node.entry.type
            self.substitutions[node] = name_node
            return name_node
        if do_visit_children:
            self.visitchildren(node)
        return node

    def visit_ExprNode(self, node):
        return self._handle_ExprNode(node, True)
    visit_Node = VisitorTransform.recurse_to_children