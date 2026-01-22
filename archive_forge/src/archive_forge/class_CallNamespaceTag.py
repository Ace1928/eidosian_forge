import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class CallNamespaceTag(Tag):

    def __init__(self, namespace, defname, attributes, **kwargs):
        super().__init__(namespace + ':' + defname, attributes, tuple(attributes.keys()) + ('args',), (), (), **kwargs)
        self.expression = '%s.%s(%s)' % (namespace, defname, ','.join(('%s=%s' % (k, v) for k, v in self.parsed_attributes.items() if k != 'args')))
        self.code = ast.PythonCode(self.expression, **self.exception_kwargs)
        self.body_decl = ast.FunctionArgs(attributes.get('args', ''), **self.exception_kwargs)

    def declared_identifiers(self):
        return self.code.declared_identifiers.union(self.body_decl.allargnames)

    def undeclared_identifiers(self):
        return self.code.undeclared_identifiers.difference(self.code.declared_identifiers)