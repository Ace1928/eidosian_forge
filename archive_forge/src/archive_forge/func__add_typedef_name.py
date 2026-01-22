from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def _add_typedef_name(self, name, coord):
    """ Add a new typedef name (ie a TYPEID) to the current scope
        """
    if not self._scope_stack[-1].get(name, True):
        self._parse_error('Typedef %r previously declared as non-typedef in this scope' % name, coord)
    self._scope_stack[-1][name] = True