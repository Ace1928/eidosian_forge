import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class DefTag(Tag):
    __keyword__ = 'def'

    def __init__(self, keyword, attributes, **kwargs):
        expressions = ['buffered', 'cached'] + [c for c in attributes if c.startswith('cache_')]
        super().__init__(keyword, attributes, expressions, ('name', 'filter', 'decorator'), ('name',), **kwargs)
        name = attributes['name']
        if re.match('^[\\w_]+$', name):
            raise exceptions.CompileException('Missing parenthesis in %def', **self.exception_kwargs)
        self.function_decl = ast.FunctionDecl('def ' + name + ':pass', **self.exception_kwargs)
        self.name = self.function_decl.funcname
        self.decorator = attributes.get('decorator', '')
        self.filter_args = ast.ArgumentList(attributes.get('filter', ''), **self.exception_kwargs)
    is_anonymous = False
    is_block = False

    @property
    def funcname(self):
        return self.function_decl.funcname

    def get_argument_expressions(self, **kw):
        return self.function_decl.get_argument_expressions(**kw)

    def declared_identifiers(self):
        return self.function_decl.allargnames

    def undeclared_identifiers(self):
        res = []
        for c in self.function_decl.defaults:
            res += list(ast.PythonCode(c, **self.exception_kwargs).undeclared_identifiers)
        return set(res).union(self.filter_args.undeclared_identifiers.difference(filters.DEFAULT_ESCAPES.keys())).union(self.expression_undeclared_identifiers).difference(self.function_decl.allargnames)