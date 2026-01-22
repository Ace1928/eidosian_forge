from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
class TemplateASTTransformer(ASTTransformer):
    """Concrete AST transformer that implements the AST transformations needed
    for code embedded in templates.
    """

    def __init__(self):
        self.locals = [CONSTANTS]

    def _process(self, names, node):
        if not IS_PYTHON2 and isinstance(node, _ast.arg):
            names.add(node.arg)
        elif isstring(node):
            names.add(node)
        elif isinstance(node, _ast.Name):
            names.add(node.id)
        elif isinstance(node, _ast.alias):
            names.add(node.asname or node.name)
        elif isinstance(node, _ast.Tuple):
            for elt in node.elts:
                self._process(names, elt)

    def _extract_names(self, node):
        names = set()
        if hasattr(node, 'args'):
            for arg in node.args:
                self._process(names, arg)
            if hasattr(node, 'kwonlyargs'):
                for arg in node.kwonlyargs:
                    self._process(names, arg)
            if hasattr(node, 'vararg'):
                self._process(names, node.vararg)
            if hasattr(node, 'kwarg'):
                self._process(names, node.kwarg)
        elif hasattr(node, 'names'):
            for elt in node.names:
                self._process(names, elt)
        return names

    def visit_Str(self, node):
        if not isinstance(node.s, six.text_type):
            try:
                node.s.decode('ascii')
            except ValueError:
                return _new(_ast_Str, node.s.decode('utf-8'))
        return node

    def visit_ClassDef(self, node):
        if len(self.locals) > 1:
            self.locals[-1].add(node.name)
        self.locals.append(set())
        try:
            return ASTTransformer.visit_ClassDef(self, node)
        finally:
            self.locals.pop()

    def visit_Import(self, node):
        if len(self.locals) > 1:
            self.locals[-1].update(self._extract_names(node))
        return ASTTransformer.visit_Import(self, node)

    def visit_ImportFrom(self, node):
        if [a.name for a in node.names] == ['*']:
            return node
        if len(self.locals) > 1:
            self.locals[-1].update(self._extract_names(node))
        return ASTTransformer.visit_ImportFrom(self, node)

    def visit_FunctionDef(self, node):
        if len(self.locals) > 1:
            self.locals[-1].add(node.name)
        self.locals.append(self._extract_names(node.args))
        try:
            return ASTTransformer.visit_FunctionDef(self, node)
        finally:
            self.locals.pop()

    def visit_GeneratorExp(self, node):
        gens = []
        for generator in node.generators:
            self.locals.append(set())
            gen = _new(_ast.comprehension, self.visit(generator.target), self.visit(generator.iter), [self.visit(if_) for if_ in generator.ifs])
            gens.append(gen)
        ret = _new(node.__class__, self.visit(node.elt), gens)
        del self.locals[-len(node.generators):]
        return ret
    visit_ListComp = visit_GeneratorExp

    def visit_Lambda(self, node):
        self.locals.append(self._extract_names(node.args))
        try:
            return ASTTransformer.visit_Lambda(self, node)
        finally:
            self.locals.pop()

    def visit_Starred(self, node):
        node.value = self.visit(node.value)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, _ast.Load) and node.id not in flatten(self.locals):
            name = _new(_ast.Name, '_lookup_name', _ast.Load())
            namearg = _new(_ast.Name, '__data__', _ast.Load())
            strarg = _new(_ast_Str, node.id)
            node = _new(_ast.Call, name, [namearg, strarg], [])
        elif isinstance(node.ctx, _ast.Store):
            if len(self.locals) > 1:
                self.locals[-1].add(node.id)
        return node