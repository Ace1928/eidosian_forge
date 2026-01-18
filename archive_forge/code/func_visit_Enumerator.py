from . import c_ast
def visit_Enumerator(self, n):
    if not n.value:
        return '{indent}{name},\n'.format(indent=self._make_indent(), name=n.name)
    else:
        return '{indent}{name} = {value},\n'.format(indent=self._make_indent(), name=n.name, value=self.visit(n.value))