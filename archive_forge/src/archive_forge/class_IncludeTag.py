import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class IncludeTag(Tag):
    __keyword__ = 'include'

    def __init__(self, keyword, attributes, **kwargs):
        super().__init__(keyword, attributes, ('file', 'import', 'args'), (), ('file',), **kwargs)
        self.page_args = ast.PythonCode('__DUMMY(%s)' % attributes.get('args', ''), **self.exception_kwargs)

    def declared_identifiers(self):
        return []

    def undeclared_identifiers(self):
        identifiers = self.page_args.undeclared_identifiers.difference({'__DUMMY'}).difference(self.page_args.declared_identifiers)
        return identifiers.union(super().undeclared_identifiers())