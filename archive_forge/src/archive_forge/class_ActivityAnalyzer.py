import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
class ActivityAnalyzer(transformer.Base):
    """Annotates nodes with local scope information.

  See Scope.

  The use of this class requires that qual_names.resolve() has been called on
  the node. This class will ignore nodes have not been
  annotated with their qualified names.
  """

    def __init__(self, context, parent_scope=None):
        super(ActivityAnalyzer, self).__init__(context)
        self.allow_skips = False
        self.scope = Scope(parent_scope, isolated=True)
        self._in_aug_assign = False
        self._in_annotation = False
        self._track_annotations_only = False

    @property
    def _in_constructor(self):
        context = self.state[_FunctionOrClass]
        if context.level > 2:
            innermost = context.stack[-1].node
            parent = context.stack[-2].node
            return isinstance(parent, gast.ClassDef) and (isinstance(innermost, gast.FunctionDef) and innermost.name == '__init__')
        return False

    def _node_sets_self_attribute(self, node):
        if anno.hasanno(node, anno.Basic.QN):
            qn = anno.getanno(node, anno.Basic.QN)
            if qn.has_attr and qn.parent.qn == ('self',):
                return True
        return False

    def _track_symbol(self, node, composite_writes_alter_parent=False):
        if self._track_annotations_only and (not self._in_annotation):
            return
        if not anno.hasanno(node, anno.Basic.QN):
            return
        qn = anno.getanno(node, anno.Basic.QN)
        for l in self.state[_Comprehension]:
            if qn in l.targets:
                return
            if qn.owner_set & set(l.targets):
                return
        if isinstance(node.ctx, gast.Store):
            if self.state[_Comprehension].level > 0:
                self.state[_Comprehension].targets.add(qn)
                return
            self.scope.modified.add(qn)
            self.scope.bound.add(qn)
            if qn.is_composite and composite_writes_alter_parent:
                self.scope.modified.add(qn.parent)
            if self._in_aug_assign:
                self.scope.read.add(qn)
        elif isinstance(node.ctx, gast.Load):
            self.scope.read.add(qn)
            if self._in_annotation:
                self.scope.annotations.add(qn)
        elif isinstance(node.ctx, gast.Param):
            self.scope.bound.add(qn)
            self.scope.mark_param(qn, self.state[_FunctionOrClass].node)
        elif isinstance(node.ctx, gast.Del):
            self.scope.read.add(qn)
            self.scope.bound.add(qn)
            self.scope.deleted.add(qn)
        else:
            raise ValueError('Unknown context {} for node "{}".'.format(type(node.ctx), qn))

    def _enter_scope(self, isolated, f_name=None):
        self.scope = Scope(self.scope, isolated=isolated, function_name=f_name)

    def _exit_scope(self):
        exited_scope = self.scope
        exited_scope.finalize()
        self.scope = exited_scope.parent
        return exited_scope

    def _exit_and_record_scope(self, node, tag=anno.Static.SCOPE):
        node_scope = self._exit_scope()
        anno.setanno(node, tag, node_scope)
        return node_scope

    def _process_statement(self, node):
        self._enter_scope(False)
        node = self.generic_visit(node)
        self._exit_and_record_scope(node)
        return node

    def _process_annotation(self, node):
        self._in_annotation = True
        node = self.visit(node)
        self._in_annotation = False
        return node

    def visit_Import(self, node):
        return self._process_statement(node)

    def visit_ImportFrom(self, node):
        return self._process_statement(node)

    def visit_Global(self, node):
        self._enter_scope(False)
        for name in node.names:
            qn = qual_names.QN(name)
            self.scope.read.add(qn)
            self.scope.globals.add(qn)
        self._exit_and_record_scope(node)
        return node

    def visit_Nonlocal(self, node):
        self._enter_scope(False)
        for name in node.names:
            qn = qual_names.QN(name)
            self.scope.read.add(qn)
            self.scope.bound.add(qn)
            self.scope.nonlocals.add(qn)
        self._exit_and_record_scope(node)
        return node

    def visit_Expr(self, node):
        return self._process_statement(node)

    def visit_Raise(self, node):
        return self._process_statement(node)

    def visit_Return(self, node):
        return self._process_statement(node)

    def visit_Assign(self, node):
        return self._process_statement(node)

    def visit_AnnAssign(self, node):
        self._enter_scope(False)
        node.target = self.visit(node.target)
        if node.value is not None:
            node.value = self.visit(node.value)
        if node.annotation:
            node.annotation = self._process_annotation(node.annotation)
        self._exit_and_record_scope(node)
        return node

    def visit_AugAssign(self, node):
        self._enter_scope(False)
        self._in_aug_assign = True
        node.target = self.visit(node.target)
        self._in_aug_assign = False
        node.op = self.visit(node.op)
        node.value = self.visit(node.value)
        self._exit_and_record_scope(node)
        return node

    def visit_Delete(self, node):
        return self._process_statement(node)

    def visit_Name(self, node):
        if node.annotation:
            node.annotation = self._process_annotation(node.annotation)
        self._track_symbol(node)
        return node

    def visit_alias(self, node):
        node = self.generic_visit(node)
        if node.asname is None:
            qn = qual_names.QN(node.name.split('.')[0])
        else:
            qn = qual_names.QN(node.asname)
        self.scope.modified.add(qn)
        self.scope.bound.add(qn)
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if self._in_constructor and self._node_sets_self_attribute(node):
            self._track_symbol(node, composite_writes_alter_parent=True)
        else:
            self._track_symbol(node)
        return node

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        self._track_symbol(node)
        return node

    def visit_Print(self, node):
        self._enter_scope(False)
        node.values = self.visit_block(node.values)
        node_scope = self._exit_and_record_scope(node)
        anno.setanno(node, NodeAnno.ARGS_SCOPE, node_scope)
        return node

    def visit_Assert(self, node):
        return self._process_statement(node)

    def visit_Call(self, node):
        self._enter_scope(False)
        node.args = self.visit_block(node.args)
        node.keywords = self.visit_block(node.keywords)
        self._exit_and_record_scope(node, tag=NodeAnno.ARGS_SCOPE)
        node.func = self.visit(node.func)
        return node

    def _process_block_node(self, node, block, scope_name):
        self._enter_scope(False)
        block = self.visit_block(block)
        self._exit_and_record_scope(node, tag=scope_name)
        return node

    def _process_parallel_blocks(self, parent, children):
        before_parent = Scope.copy_of(self.scope)
        after_children = []
        for child, scope_name in children:
            self.scope.copy_from(before_parent)
            parent = self._process_block_node(parent, child, scope_name)
            after_child = Scope.copy_of(self.scope)
            after_children.append(after_child)
        for after_child in after_children:
            self.scope.merge_from(after_child)
        return parent

    def _process_comprehension(self, node, is_list_comp=False, is_dict_comp=False):
        with self.state[_Comprehension] as comprehension_:
            comprehension_.is_list_comp = is_list_comp
            node.generators = self.visit_block(node.generators)
            if is_dict_comp:
                node.key = self.visit(node.key)
                node.value = self.visit(node.value)
            else:
                node.elt = self.visit(node.elt)
            return node

    def visit_comprehension(self, node):
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)
        return self.generic_visit(node)

    def visit_DictComp(self, node):
        return self._process_comprehension(node, is_dict_comp=True)

    def visit_ListComp(self, node):
        return self._process_comprehension(node, is_list_comp=True)

    def visit_SetComp(self, node):
        return self._process_comprehension(node)

    def visit_GeneratorExp(self, node):
        return self._process_comprehension(node)

    def visit_ClassDef(self, node):
        with self.state[_FunctionOrClass] as fn:
            fn.node = node
            self._enter_scope(False)
            node.decorator_list = self.visit_block(node.decorator_list)
            self.scope.modified.add(qual_names.QN(node.name))
            self.scope.bound.add(qual_names.QN(node.name))
            node.bases = self.visit_block(node.bases)
            node.keywords = self.visit_block(node.keywords)
            self._exit_and_record_scope(node)
            self._enter_scope(True)
            node = self.generic_visit(node)
            self._exit_scope()
            return node

    def _visit_node_list(self, nodes):
        return [None if n is None else self.visit(n) for n in nodes]

    def _visit_arg_annotations(self, node):
        node.args.kw_defaults = self._visit_node_list(node.args.kw_defaults)
        node.args.defaults = self._visit_node_list(node.args.defaults)
        self._track_annotations_only = True
        node = self._visit_arg_declarations(node)
        self._track_annotations_only = False
        return node

    def _visit_arg_declarations(self, node):
        node.args.posonlyargs = self._visit_node_list(node.args.posonlyargs)
        node.args.args = self._visit_node_list(node.args.args)
        if node.args.vararg is not None:
            node.args.vararg = self.visit(node.args.vararg)
        node.args.kwonlyargs = self._visit_node_list(node.args.kwonlyargs)
        if node.args.kwarg is not None:
            node.args.kwarg = self.visit(node.args.kwarg)
        return node

    def visit_FunctionDef(self, node):
        with self.state[_FunctionOrClass] as fn:
            fn.node = node
            self._enter_scope(False)
            node.decorator_list = self.visit_block(node.decorator_list)
            if node.returns:
                node.returns = self._process_annotation(node.returns)
            node = self._visit_arg_annotations(node)
            function_name = qual_names.QN(node.name)
            self.scope.modified.add(function_name)
            self.scope.bound.add(function_name)
            self._exit_and_record_scope(node)
            self._enter_scope(True, node.name)
            self._enter_scope(False, node.name)
            node = self._visit_arg_declarations(node)
            self._exit_and_record_scope(node.args)
            self._enter_scope(False, node.name)
            node.body = self.visit_block(node.body)
            self._exit_and_record_scope(node, NodeAnno.BODY_SCOPE)
            self._exit_and_record_scope(node, NodeAnno.ARGS_AND_BODY_SCOPE)
            return node

    def visit_Lambda(self, node):
        with self.state[_FunctionOrClass] as fn:
            fn.node = node
            self._enter_scope(False)
            node = self._visit_arg_annotations(node)
            self._exit_and_record_scope(node)
            self._enter_scope(True)
            self._enter_scope(False)
            node = self._visit_arg_declarations(node)
            self._exit_and_record_scope(node.args)
            self._enter_scope(False)
            node.body = self.visit(node.body)
            if not anno.hasanno(node.body, anno.Static.SCOPE):
                anno.setanno(node.body, anno.Static.SCOPE, self.scope)
            self._exit_and_record_scope(node, NodeAnno.BODY_SCOPE)
            lambda_scope = self.scope
            self._exit_and_record_scope(node, NodeAnno.ARGS_AND_BODY_SCOPE)
            self.scope.read.update(lambda_scope.read - lambda_scope.bound)
            return node

    def visit_With(self, node):
        self._enter_scope(False)
        node = self.generic_visit(node)
        self._exit_and_record_scope(node, NodeAnno.BODY_SCOPE)
        return node

    def visit_withitem(self, node):
        return self._process_statement(node)

    def visit_If(self, node):
        self._enter_scope(False)
        node.test = self.visit(node.test)
        node_scope = self._exit_and_record_scope(node.test)
        anno.setanno(node, NodeAnno.COND_SCOPE, node_scope)
        node = self._process_parallel_blocks(node, ((node.body, NodeAnno.BODY_SCOPE), (node.orelse, NodeAnno.ORELSE_SCOPE)))
        return node

    def visit_For(self, node):
        self._enter_scope(False)
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        self._exit_and_record_scope(node.iter)
        self._enter_scope(False)
        self.visit(node.target)
        if anno.hasanno(node, anno.Basic.EXTRA_LOOP_TEST):
            self._process_statement(anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST))
        self._exit_and_record_scope(node, tag=NodeAnno.ITERATE_SCOPE)
        node = self._process_parallel_blocks(node, ((node.body, NodeAnno.BODY_SCOPE), (node.orelse, NodeAnno.ORELSE_SCOPE)))
        return node

    def visit_While(self, node):
        self._enter_scope(False)
        node.test = self.visit(node.test)
        node_scope = self._exit_and_record_scope(node.test)
        anno.setanno(node, NodeAnno.COND_SCOPE, node_scope)
        node = self._process_parallel_blocks(node, ((node.body, NodeAnno.BODY_SCOPE), (node.orelse, NodeAnno.ORELSE_SCOPE)))
        return node

    def visit_ExceptHandler(self, node):
        self._enter_scope(False)
        if node.name is not None:
            self.scope.isolated_names.add(anno.getanno(node.name, anno.Basic.QN))
        node = self.generic_visit(node)
        self._exit_scope()
        return node