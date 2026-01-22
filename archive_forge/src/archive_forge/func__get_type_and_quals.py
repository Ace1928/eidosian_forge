from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _get_type_and_quals(self, typenode, name=None, partial_length_ok=False, typedef_example=None):
    if isinstance(typenode, pycparser.c_ast.TypeDecl) and isinstance(typenode.type, pycparser.c_ast.IdentifierType) and (len(typenode.type.names) == 1) and ('typedef ' + typenode.type.names[0] in self._declarations):
        tp, quals = self._declarations['typedef ' + typenode.type.names[0]]
        quals |= self._extract_quals(typenode)
        return (tp, quals)
    if isinstance(typenode, pycparser.c_ast.ArrayDecl):
        if typenode.dim is None:
            length = None
        else:
            length = self._parse_constant(typenode.dim, partial_length_ok=partial_length_ok)
        if typedef_example is not None:
            if length == '...':
                length = '_cffi_array_len(%s)' % (typedef_example,)
            typedef_example = '*' + typedef_example
        tp, quals = self._get_type_and_quals(typenode.type, partial_length_ok=partial_length_ok, typedef_example=typedef_example)
        return (model.ArrayType(tp, length), quals)
    if isinstance(typenode, pycparser.c_ast.PtrDecl):
        itemtype, itemquals = self._get_type_and_quals(typenode.type)
        tp = self._get_type_pointer(itemtype, itemquals, declname=name)
        quals = self._extract_quals(typenode)
        return (tp, quals)
    if isinstance(typenode, pycparser.c_ast.TypeDecl):
        quals = self._extract_quals(typenode)
        type = typenode.type
        if isinstance(type, pycparser.c_ast.IdentifierType):
            names = list(type.names)
            if names != ['signed', 'char']:
                prefixes = {}
                while names:
                    name = names[0]
                    if name in ('short', 'long', 'signed', 'unsigned'):
                        prefixes[name] = prefixes.get(name, 0) + 1
                        del names[0]
                    else:
                        break
                newnames = []
                for prefix in ('unsigned', 'short', 'long'):
                    for i in range(prefixes.get(prefix, 0)):
                        newnames.append(prefix)
                if not names:
                    names = ['int']
                if names == ['int']:
                    if 'short' in prefixes or 'long' in prefixes:
                        names = []
                names = newnames + names
            ident = ' '.join(names)
            if ident == 'void':
                return (model.void_type, quals)
            if ident == '__dotdotdot__':
                raise FFIError(':%d: bad usage of "..."' % typenode.coord.line)
            tp0, quals0 = resolve_common_type(self, ident)
            return (tp0, quals | quals0)
        if isinstance(type, pycparser.c_ast.Struct):
            tp = self._get_struct_union_enum_type('struct', type, name)
            return (tp, quals)
        if isinstance(type, pycparser.c_ast.Union):
            tp = self._get_struct_union_enum_type('union', type, name)
            return (tp, quals)
        if isinstance(type, pycparser.c_ast.Enum):
            tp = self._get_struct_union_enum_type('enum', type, name)
            return (tp, quals)
    if isinstance(typenode, pycparser.c_ast.FuncDecl):
        return (self._parse_function_type(typenode, name), 0)
    if isinstance(typenode, pycparser.c_ast.Struct):
        return (self._get_struct_union_enum_type('struct', typenode, name, nested=True), 0)
    if isinstance(typenode, pycparser.c_ast.Union):
        return (self._get_struct_union_enum_type('union', typenode, name, nested=True), 0)
    raise FFIError(':%d: bad or unsupported type declaration' % typenode.coord.line)