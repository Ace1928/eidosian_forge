from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def _fix_decl_name_type(self, decl, typename):
    """ Fixes a declaration. Modifies decl.
        """
    type = decl
    while not isinstance(type, c_ast.TypeDecl):
        type = type.type
    decl.name = type.declname
    type.quals = decl.quals[:]
    for tn in typename:
        if not isinstance(tn, c_ast.IdentifierType):
            if len(typename) > 1:
                self._parse_error('Invalid multiple types specified', tn.coord)
            else:
                type.type = tn
                return decl
    if not typename:
        if not isinstance(decl.type, c_ast.FuncDecl):
            self._parse_error('Missing type in declaration', decl.coord)
        type.type = c_ast.IdentifierType(['int'], coord=decl.coord)
    else:
        type.type = c_ast.IdentifierType([name for id in typename for name in id.names], coord=typename[0].coord)
    return decl