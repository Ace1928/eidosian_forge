from . import c_ast
def visit_Enum(self, n):
    return self._generate_struct_union_enum(n, name='enum')