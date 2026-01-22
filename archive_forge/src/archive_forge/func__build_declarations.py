from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def _build_declarations(self, spec, decls, typedef_namespace=False):
    """ Builds a list of declarations all sharing the given specifiers.
            If typedef_namespace is true, each declared name is added
            to the "typedef namespace", which also includes objects,
            functions, and enum constants.
        """
    is_typedef = 'typedef' in spec['storage']
    declarations = []
    if decls[0].get('bitsize') is not None:
        pass
    elif decls[0]['decl'] is None:
        if len(spec['type']) < 2 or len(spec['type'][-1].names) != 1 or (not self._is_type_in_scope(spec['type'][-1].names[0])):
            coord = '?'
            for t in spec['type']:
                if hasattr(t, 'coord'):
                    coord = t.coord
                    break
            self._parse_error('Invalid declaration', coord)
        decls[0]['decl'] = c_ast.TypeDecl(declname=spec['type'][-1].names[0], type=None, quals=None, align=spec['alignment'], coord=spec['type'][-1].coord)
        del spec['type'][-1]
    elif not isinstance(decls[0]['decl'], (c_ast.Enum, c_ast.Struct, c_ast.Union, c_ast.IdentifierType)):
        decls_0_tail = decls[0]['decl']
        while not isinstance(decls_0_tail, c_ast.TypeDecl):
            decls_0_tail = decls_0_tail.type
        if decls_0_tail.declname is None:
            decls_0_tail.declname = spec['type'][-1].names[0]
            del spec['type'][-1]
    for decl in decls:
        assert decl['decl'] is not None
        if is_typedef:
            declaration = c_ast.Typedef(name=None, quals=spec['qual'], storage=spec['storage'], type=decl['decl'], coord=decl['decl'].coord)
        else:
            declaration = c_ast.Decl(name=None, quals=spec['qual'], align=spec['alignment'], storage=spec['storage'], funcspec=spec['function'], type=decl['decl'], init=decl.get('init'), bitsize=decl.get('bitsize'), coord=decl['decl'].coord)
        if isinstance(declaration.type, (c_ast.Enum, c_ast.Struct, c_ast.Union, c_ast.IdentifierType)):
            fixed_decl = declaration
        else:
            fixed_decl = self._fix_decl_name_type(declaration, spec['type'])
        if typedef_namespace:
            if is_typedef:
                self._add_typedef_name(fixed_decl.name, fixed_decl.coord)
            else:
                self._add_identifier(fixed_decl.name, fixed_decl.coord)
        fixed_decl = fix_atomic_specifiers(fixed_decl)
        declarations.append(fixed_decl)
    return declarations