from . import c_ast
def visit_Union(self, n):
    return self._generate_struct_union_enum(n, 'union')