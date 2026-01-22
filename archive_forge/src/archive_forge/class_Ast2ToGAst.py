from astn import AstToGAst, GAstToAst
import ast
import gast
class Ast2ToGAst(AstToGAst):

    def visit_Module(self, node):
        new_node = gast.Module(self._visit(node.body), [])
        return new_node

    def visit_FunctionDef(self, node):
        new_node = gast.FunctionDef(self._visit(node.name), self._visit(node.args), self._visit(node.body), self._visit(node.decorator_list), None, None)
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_ClassDef(self, node):
        new_node = gast.ClassDef(self._visit(node.name), self._visit(node.bases), [], self._visit(node.body), self._visit(node.decorator_list))
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_Assign(self, node):
        new_node = gast.Assign(self._visit(node.targets), self._visit(node.value), None)
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_For(self, node):
        new_node = gast.For(self._visit(node.target), self._visit(node.iter), self._visit(node.body), self._visit(node.orelse), [])
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_With(self, node):
        new_node = gast.With([gast.withitem(self._visit(node.context_expr), self._visit(node.optional_vars))], self._visit(node.body), None)
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_Raise(self, node):
        ntype = self._visit(node.type)
        ninst = self._visit(node.inst)
        ntback = self._visit(node.tback)
        what = ntype
        if ninst is not None:
            what = gast.Call(ntype, [ninst], [])
            gast.copy_location(what, node)
            what.end_lineno = what.end_col_offset = None
        if ntback is not None:
            attr = gast.Attribute(what, 'with_traceback', gast.Load())
            gast.copy_location(attr, node)
            attr.end_lineno = attr.end_col_offset = None
            what = gast.Call(attr, [ntback], [])
            gast.copy_location(what, node)
            what.end_lineno = what.end_col_offset = None
        new_node = gast.Raise(what, None)
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_TryExcept(self, node):
        new_node = gast.Try(self._visit(node.body), self._visit(node.handlers), self._visit(node.orelse), [])
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_TryFinally(self, node):
        new_node = gast.Try(self._visit(node.body), [], [], self._visit(node.finalbody))
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_Name(self, node):
        new_node = gast.Name(self._visit(node.id), self._visit(node.ctx), None, None)
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_Num(self, node):
        new_node = gast.Constant(node.n, None)
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_Subscript(self, node):
        new_slice = self._visit(node.slice)
        new_node = gast.Subscript(self._visit(node.value), new_slice, self._visit(node.ctx))
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_Ellipsis(self, node):
        new_node = gast.Constant(Ellipsis, None)
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_Index(self, node):
        return self._visit(node.value)

    def visit_ExtSlice(self, node):
        new_dims = self._visit(node.dims)
        new_node = gast.Tuple(new_dims, gast.Load())
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_Str(self, node):
        new_node = gast.Constant(node.s, None)
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_Call(self, node):
        if node.starargs:
            star = gast.Starred(self._visit(node.starargs), gast.Load())
            gast.copy_location(star, node)
            star.end_lineno = star.end_col_offset = None
            starred = [star]
        else:
            starred = []
        if node.kwargs:
            kwargs = [gast.keyword(None, self._visit(node.kwargs))]
        else:
            kwargs = []
        new_node = gast.Call(self._visit(node.func), self._visit(node.args) + starred, self._visit(node.keywords) + kwargs)
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_comprehension(self, node):
        new_node = gast.comprehension(target=self._visit(node.target), iter=self._visit(node.iter), ifs=self._visit(node.ifs), is_async=0)
        gast.copy_location(new_node, node)
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node

    def visit_arguments(self, node):
        if node.vararg:
            vararg = ast.Name(node.vararg, ast.Param())
        else:
            vararg = None
        if node.kwarg:
            kwarg = ast.Name(node.kwarg, ast.Param())
        else:
            kwarg = None
        if node.vararg:
            vararg = ast.Name(node.vararg, ast.Param())
        else:
            vararg = None
        new_node = gast.arguments(self._visit(node.args), [], self._visit(vararg), [], [], self._visit(kwarg), self._visit(node.defaults))
        return new_node

    def visit_alias(self, node):
        new_node = gast.alias(self._visit(node.name), self._visit(node.asname))
        new_node.lineno = new_node.col_offset = None
        new_node.end_lineno = new_node.end_col_offset = None
        return new_node