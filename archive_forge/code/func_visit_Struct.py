from . import c_ast
def visit_Struct(self, n):
    return self._generate_struct_union_enum(n, 'struct')