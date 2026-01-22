import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class ControlLine(Node):
    """defines a control line, a line-oriented python line or end tag.

    e.g.::

        % if foo:
            (markup)
        % endif

    """
    has_loop_context = False

    def __init__(self, keyword, isend, text, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.keyword = keyword
        self.isend = isend
        self.is_primary = keyword in ['for', 'if', 'while', 'try', 'with']
        self.nodes = []
        if self.isend:
            self._declared_identifiers = []
            self._undeclared_identifiers = []
        else:
            code = ast.PythonFragment(text, **self.exception_kwargs)
            self._declared_identifiers = code.declared_identifiers
            self._undeclared_identifiers = code.undeclared_identifiers

    def get_children(self):
        return self.nodes

    def declared_identifiers(self):
        return self._declared_identifiers

    def undeclared_identifiers(self):
        return self._undeclared_identifiers

    def is_ternary(self, keyword):
        """return true if the given keyword is a ternary keyword
        for this ControlLine"""
        cases = {'if': {'else', 'elif'}, 'try': {'except', 'finally'}, 'for': {'else'}}
        return keyword in cases.get(self.keyword, set())

    def __repr__(self):
        return 'ControlLine(%r, %r, %r, %r)' % (self.keyword, self.text, self.isend, (self.lineno, self.pos))