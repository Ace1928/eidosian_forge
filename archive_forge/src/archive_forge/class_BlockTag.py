import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class BlockTag(Tag):
    __keyword__ = 'block'

    def __init__(self, keyword, attributes, **kwargs):
        expressions = ['buffered', 'cached', 'args'] + [c for c in attributes if c.startswith('cache_')]
        super().__init__(keyword, attributes, expressions, ('name', 'filter', 'decorator'), (), **kwargs)
        name = attributes.get('name')
        if name and (not re.match('^[\\w_]+$', name)):
            raise exceptions.CompileException('%block may not specify an argument signature', **self.exception_kwargs)
        if not name and attributes.get('args', None):
            raise exceptions.CompileException('Only named %blocks may specify args', **self.exception_kwargs)
        self.body_decl = ast.FunctionArgs(attributes.get('args', ''), **self.exception_kwargs)
        self.name = name
        self.decorator = attributes.get('decorator', '')
        self.filter_args = ast.ArgumentList(attributes.get('filter', ''), **self.exception_kwargs)
    is_block = True

    @property
    def is_anonymous(self):
        return self.name is None

    @property
    def funcname(self):
        return self.name or '__M_anon_%d' % (self.lineno,)

    def get_argument_expressions(self, **kw):
        return self.body_decl.get_argument_expressions(**kw)

    def declared_identifiers(self):
        return self.body_decl.allargnames

    def undeclared_identifiers(self):
        return self.filter_args.undeclared_identifiers.difference(filters.DEFAULT_ESCAPES.keys()).union(self.expression_undeclared_identifiers)